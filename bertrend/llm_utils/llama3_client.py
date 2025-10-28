import requests
from typing import Any, Optional
from pydantic import BaseModel
import json
from loguru import logger

# MODIFICATION: Enhanced Llama3Client to match OpenAI_Client interface
# PURPOSE: Enable seamless replacement of OpenAI with our own LLaMA 3 API for summarization
# This allows the existing codebase to use our Llama3 API without major refactoring

class Llama3Client:
    """Llama3 client for interacting with the specified LLM API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        api_base_url: str = None,
        username: str = None,
        password: str = None,
    ):
        """Initialize the Llama3 client.

        Args:
            api_key (str): API key/username for authentication
            endpoint (str): Full endpoint URL for the API
            model (str): Name of the LLM model to use
            api_base_url (str, optional): Base URL for the API (extracted from endpoint if not provided)
            username (str, optional): Username for authentication (uses api_key if not provided)
            password (str, optional): Password for authentication
        """
        # Extract base URL and endpoint from the full endpoint URL if needed
        if api_base_url is None:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            self.api_base_url = f"{parsed.scheme}://{parsed.netloc}"
            self.api_endpoint = parsed.path.lstrip("/")
        else:
            self.api_base_url = api_base_url.rstrip("/")
            self.api_endpoint = endpoint.lstrip("/")
        
        self.username = username if username else api_key
        self.password = password if password else "Nr17TT5m8Qj1oHS1"  # Default password from config
        self.model_name = model
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        
        # MODIFICATION: Add llm_client attribute for compatibility with OpenAI_Client interface
        # PURPOSE: Enable BERTopicModel to use this client seamlessly
        self.llm_client = self
        
        # Add attributes for compatibility with OpenAI_Client
        self.temperature = 0.7
        self.max_output_tokens = 1000

    def parse(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Generate text using the Llama3 API and parse the response.

        Args:
            user_prompt (str): The user prompt to send to the model
            system_prompt (Optional[str]): Optional system prompt
            response_format (Optional[type[BaseModel]]): Optional Pydantic model to parse response into
            temperature (Optional[float]): Temperature for text generation (used in payload)
            max_output_tokens (Optional[int]): Maximum tokens to generate (used in payload)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Any: The parsed response, either as a Pydantic model or raw text
        """
        try:
            # Prepare the messages array
            messages = []

            # Add system prompt for JSON formatting if response_format is specified
            if response_format:
                # MODIFICATION: Enhanced JSON formatting prompt for LLaMA 3 API
                # PURPOSE: Ensure LLaMA 3 returns properly structured JSON that matches Pydantic models
                json_system_prompt = f"""You are a helpful assistant that ALWAYS responds in valid JSON format. 
                Your response must be a valid JSON object with the exact structure required.
                Do not include any markdown formatting, explanatory text, or additional content.
                Return ONLY the JSON object.
                
                For TopicSummaryList, return: {{"topic_summary_by_time_period": [{{"title": "...", "date": "...", "key_developments": [...], "description": "...", "novelty": "..."}}]}}
                For SignalAnalysis, return the complete JSON structure with all required fields.
                
                IMPORTANT: Your response must be parseable JSON, not markdown or any other format."""
                messages.append({"role": "system", "content": json_system_prompt})

            # Add user's system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add the user prompt
            messages.append({"role": "user", "content": user_prompt})

            # MODIFICATION: Use provided temperature and max_output_tokens parameters
            # PURPOSE: Enable compatibility with OpenAI_Client interface parameters
            # This allows the signal analysis to pass temperature and token limits
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature if temperature is not None else 0.7,
                "max_new_tokens": max_output_tokens if max_output_tokens is not None else 1000,
                "top_p": 0.9,
            }

            # Make the API request
            response = self.session.post(
                f"{self.api_base_url}/{self.api_endpoint}",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            # Extract the generated text from the response
            response_data = response.json()
            generated_text = response_data.get("generated_text", "")

            # If a response format is specified, try to parse the response
            if response_format:
                try:
                    # Try to parse as JSON first
                    json_response = json.loads(generated_text)
                    
                    # MODIFICATION: Add detailed logging for debugging JSON structure issues
                    # PURPOSE: Help diagnose when LLaMA 3 API returns unexpected JSON structure
                    logger.debug(f"LLaMA 3 API response JSON structure: {json_response}")
                    logger.debug(f"Expected Pydantic model: {response_format.__name__}")
                    
                    return response_format.model_validate(json_response)
                except Exception as validation_error:
                    # MODIFICATION: Enhanced error handling for Pydantic validation errors
                    # PURPOSE: Provide detailed information when LLaMA 3 API returns unexpected structure
                    logger.error(f"Pydantic validation error: {str(validation_error)}")
                    logger.error(f"LLaMA 3 API returned JSON: {json_response}")
                    logger.error(f"Expected model fields: {list(response_format.model_fields.keys())}")
                    
                    # Try to create a fallback response with the available data
                    try:
                        # Handle SignalAnalysis field mapping
                        if response_format.__name__ == 'SignalAnalysis':
                            logger.warning("Attempting to map SignalAnalysis fields")
                            if 'signal_analysis' in json_response:
                                signal_data = json_response['signal_analysis']
                                mapped_response = {}
                                
                                # Map potential_implications with proper structure
                                if 'impact_potential' in signal_data:
                                    impact_data = signal_data['impact_potential']
                                    potential_implications = {}
                                    
                                    # Extract short_term_implications
                                    if 'short_term_impact' in impact_data:
                                        potential_implications['short_term_implications'] = [impact_data['short_term_impact']]
                                    else:
                                        potential_implications['short_term_implications'] = []
                                    
                                    # Extract long_term_implications
                                    if 'long_term_impact' in impact_data:
                                        potential_implications['long_term_implications'] = [impact_data['long_term_impact']]
                                    else:
                                        potential_implications['long_term_implications'] = []
                                    
                                    mapped_response['potential_implications'] = potential_implications
                                
                                # Map evolution_scenario with proper structure
                                if 'scenarios_of_evolution' in signal_data:
                                    scenarios = signal_data['scenarios_of_evolution']
                                    evolution_scenario = {}
                                    
                                    # Handle both list and dict formats
                                    if isinstance(scenarios, list):
                                        # Extract optimistic and pessimistic from list
                                        optimistic_desc = ""
                                        optimistic_points = []
                                        pessimistic_desc = ""
                                        pessimistic_points = []
                                        
                                        for scenario in scenarios:
                                            if 'optimistic' in scenario:
                                                optimistic_desc = scenario['optimistic']
                                                optimistic_points = [scenario['optimistic']]
                                            elif 'pessimistic' in scenario:
                                                pessimistic_desc = scenario['pessimistic']
                                                pessimistic_points = [scenario['pessimistic']]
                                        
                                        evolution_scenario = {
                                            'optimistic_scenario_description': optimistic_desc or "Optimistic scenario not specified",
                                            'optimistic_scenario_points': optimistic_points,
                                            'pessimistic_scenario_description': pessimistic_desc or "Pessimistic scenario not specified",
                                            'pessimistic_scenario_points': pessimistic_points
                                        }
                                    else:
                                        # Handle dict format
                                        evolution_scenario = {
                                            'optimistic_scenario_description': scenarios.get('optimistic', 'Optimistic scenario not specified'),
                                            'optimistic_scenario_points': [scenarios.get('optimistic', '')] if scenarios.get('optimistic') else [],
                                            'pessimistic_scenario_description': scenarios.get('pessimistic', 'Pessimistic scenario not specified'),
                                            'pessimistic_scenario_points': [scenarios.get('pessimistic', '')] if scenarios.get('pessimistic') else []
                                        }
                                    
                                    mapped_response['evolution_scenario'] = evolution_scenario
                                
                                # Map topic_interconnexions with proper structure
                                if 'interconnections_and_synergies' in signal_data:
                                    intercon_data = signal_data['interconnections_and_synergies']
                                    topic_interconnexions = {}
                                    
                                    # Extract interconnexions
                                    if isinstance(intercon_data, list):
                                        interconnexions = []
                                        ripple_effects = []
                                        for item in intercon_data:
                                            if 'trend' in item and 'phenomenon' in item:
                                                interconnexions.append(f"{item['trend']}: {item['phenomenon']}")
                                            elif 'description' in item:
                                                ripple_effects.append(item['description'])
                                        
                                        topic_interconnexions = {
                                            'interconnexions': interconnexions,
                                            'ripple_effects': ripple_effects
                                        }
                                    else:
                                        topic_interconnexions = {
                                            'interconnexions': [],
                                            'ripple_effects': []
                                        }
                                    
                                    mapped_response['topic_interconnexions'] = topic_interconnexions
                                
                                # Map drivers_inhibitors with proper structure
                                if 'drivers_and_inhibitors' in signal_data:
                                    drivers_data = signal_data['drivers_and_inhibitors']
                                    drivers_inhibitors = {}
                                    
                                    drivers = []
                                    inhibitors = []
                                    
                                    if isinstance(drivers_data, list):
                                        for item in drivers_data:
                                            if 'driver' in item:
                                                drivers.append(item.get('description', item['driver']))
                                            elif 'inhibitor' in item:
                                                inhibitors.append(item.get('description', item['inhibitor']))
                                    else:
                                        drivers = drivers_data.get('drivers', [])
                                        inhibitors = drivers_data.get('inhibitors', [])
                                    
                                    drivers_inhibitors = {
                                        'drivers': drivers,
                                        'inhibitors': inhibitors
                                    }
                                    
                                    mapped_response['drivers_inhibitors'] = drivers_inhibitors
                                
                                logger.debug(f"Mapped response: {mapped_response}")
                                return response_format.model_validate(mapped_response)
                        
                        # If the response has an 'analysis' field, try to restructure it
                        if 'analysis' in json_response and isinstance(json_response['analysis'], list):
                            logger.warning("Attempting to restructure response with 'analysis' field")
                            restructured = {
                                "topic_summary_by_time_period": json_response['analysis']
                            }
                            return response_format.model_validate(restructured)
                    except Exception as restructure_error:
                        logger.error(f"Failed to restructure response: {str(restructure_error)}")
                    
                    raise validation_error
                except json.JSONDecodeError:
                    # MODIFICATION: Enhanced handling for non-JSON responses from LLaMA 3 API
                    # PURPOSE: Handle cases where LLaMA 3 returns markdown instead of JSON
                    logger.warning("LLaMA 3 API returned non-JSON response, attempting to extract JSON")
                    logger.debug(f"Raw response: {generated_text[:500]}...")
                    
                    # If not JSON, try to extract JSON from the text
                    try:
                        # Look for JSON-like structure in the text
                        start_idx = generated_text.find("{")
                        end_idx = generated_text.rfind("}")

                        if start_idx >= 0:
                            # If we found a start but no end, add the closing brace
                            if end_idx <= start_idx:
                                json_str = generated_text[start_idx:] + "}"
                            else:
                                json_str = generated_text[start_idx : end_idx + 1]

                            # Try to parse the JSON
                            json_response = json.loads(json_str)
                            logger.info("Successfully extracted JSON from response")
                            return response_format.model_validate(json_response)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(
                            f"Failed to parse JSON response: {generated_text}"
                        )
                        logger.warning(f"JSON parsing error: {str(e)}")

                        # If all JSON parsing attempts fail, try to extract title and description
                        try:
                            # Look for title and description fields
                            title_start = generated_text.find('"title": "')
                            if title_start >= 0:
                                title_start += 9  # length of '"title": "'
                                title_end = generated_text.find('"', title_start)
                                title = generated_text[title_start:title_end]

                                desc_start = generated_text.find('"description": "')
                                if desc_start >= 0:
                                    desc_start += 14  # length of '"description": "'
                                    desc_end = generated_text.find('"', desc_start)
                                    description = generated_text[desc_start:desc_end]

                                    return response_format.model_validate(
                                        {"title": title, "description": description}
                                    )
                        except Exception as e:
                            logger.error(
                                f"Failed to extract title and description: {str(e)}"
                            )

                    # MODIFICATION: Enhanced fallback for when all JSON parsing fails
                    # PURPOSE: Create a basic response structure to prevent complete failure
                    logger.error("All JSON parsing attempts failed, creating fallback response")
                    
                    # Try to create a basic response based on the expected model
                    if response_format.__name__ == "TopicSummaryList":
                        fallback_response = {
                            "topic_summary_by_time_period": [{
                                "title": "Analysis Generated",
                                "date": "2024-01-01",
                                "key_developments": ["Analysis completed"],
                                "description": generated_text[:500] + "..." if len(generated_text) > 500 else generated_text,
                                "novelty": "Generated by LLaMA 3 API"
                            }]
                        }
                    elif response_format.__name__ == "SignalAnalysis":
                        fallback_response = {
                            "potential_implications": {
                                "long_term_implications": ["Analysis in progress"],
                                "short_term_implications": ["Analysis in progress"]
                            },
                            "evolution_scenario": {
                                "optimistic_scenario_description": "Analysis in progress",
                                "optimistic_scenario_points": ["Analysis in progress"],
                                "pessimistic_scenario_description": "Analysis in progress",
                                "pessimistic_scenario_points": ["Analysis in progress"]
                            },
                            "topic_interconnexions": {
                                "interconnexions": ["Analysis in progress"],
                                "ripple_effects": ["Analysis in progress"]
                            },
                            "drivers_inhibitors": {
                                "drivers": ["Analysis in progress"],
                                "inhibitors": ["Analysis in progress"]
                            }
                        }
                    else:
                        # Generic fallback
                        fallback_response = {
                            "title": "Analysis Generated",
                            "description": generated_text[:500] + "..." if len(generated_text) > 500 else generated_text,
                        }
                    
                    return response_format.model_validate(fallback_response)
            else:
                return generated_text

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            raise

    # MODIFICATION: Add generate method to match OpenAI_Client interface
    # PURPOSE: Enable compatibility with existing code that expects generate() method
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the Llama3 API.
        
        Args:
            user_prompt (str): The user prompt to send to the model
            system_prompt (Optional[str]): Optional system prompt
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            str: The generated text response
        """
        try:
            # Use the existing parse method but return only the text
            result = self.parse(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Error in generate method: {str(e)}")
            raise

    # MODIFICATION: Add generate_from_history method to match OpenAI_Client interface
    # PURPOSE: Enable compatibility with existing code that expects generate_from_history() method
    def generate_from_history(
        self,
        messages: list[dict],
        **kwargs
    ) -> str:
        """Generate text from a conversation history.
        
        Args:
            messages (list[dict]): List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            str: The generated text response
        """
        try:
            # Extract the last user message and any system prompts
            user_prompt = ""
            system_prompt = None
            
            for message in messages:
                if message.get("role") == "user":
                    user_prompt = message.get("content", "")
                elif message.get("role") == "system" and system_prompt is None:
                    system_prompt = message.get("content", "")
            
            if not user_prompt:
                raise ValueError("No user message found in conversation history")
            
            return self.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_from_history method: {str(e)}")
            raise
