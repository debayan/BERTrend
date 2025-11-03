#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
from datetime import datetime
from pathlib import Path

from jinja2 import Template, Environment, FileSystemLoader
from loguru import logger

from bertrend import OUTPUT_PATH
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis

# Global variables for prompts
SIGNAL_INTRO = {
    "en": """As an elite strategic foresight analyst with extensive expertise across multiple domains and industries, your task is to conduct a comprehensive evaluation of a potential signal derived from the following topic summary:

{summary_from_first_prompt}

Leverage your knowledge and analytical skills to provide an in-depth analysis of this signal's potential impact and evolution:
""",
    "fr": """En tant qu'analyste de prospective stratégique d'élite avec une expertise étendue dans de multiples domaines et industries, votre tâche est de mener une évaluation complète d'un signal potentiel dérivé du résumé de sujet suivant :

{summary_from_first_prompt}

Utilisez vos connaissances et compétences analytiques pour fournir une analyse approfondie de l'impact potentiel et de l'évolution de ce signal :
""",
    "de": """Als Elite-Strategieprognose-Analyst mit umfangreicher Expertise in verschiedenen Bereichen und Branchen ist es Ihre Aufgabe, eine umfassende Bewertung eines potenziellen Signals durchzuführen, das aus der folgenden Themenzusammenfassung abgeleitet wurde:

{summary_from_first_prompt}

Nutzen Sie Ihr Wissen und Ihre analytischen Fähigkeiten, um eine tiefgreifende Analyse der potenziellen Auswirkungen und Entwicklung dieses Signals zu liefern:
""",
}

SIGNAL_INSTRUCTIONS = {
    "en": """
1. Potential Impact Analysis:
   - Examine the potential effects of this signal on various sectors, industries, and societal aspects.
   - Consider both short-term and long-term implications.
   - Analyze possible ripple effects and second-order consequences.

2. Evolution Scenarios:
   - Describe potential ways this signal could develop or manifest in the future.
   - Consider various factors that could influence its trajectory.
   - Explore both optimistic and pessimistic scenarios.

3. Interconnections and Synergies:
   - Identify how this signal might interact with other current trends or emerging phenomena.
   - Discuss potential synergies or conflicts with existing systems or paradigms.

4. Drivers and Inhibitors:
   - Analyze factors that could accelerate or amplify this signal.
   - Examine potential barriers or resistances that might hinder its development.

Your analysis should be thorough and nuanced, going beyond surface-level observations. Draw upon your expertise to provide insights that capture the complexity and potential significance of this signal. Don't hesitate to make well-reasoned predictions about its potential trajectory and impact.

Focus on providing a clear, insightful, and actionable analysis that can inform strategic decision-making and future planning.

Return ONLY valid JSON using the following canonical schema keys (keys must remain in English, values in the user language are fine). If information is missing, return empty lists for those fields. Do not add extra fields or prose outside JSON.

{
  "potential_implications": {
    "short_term_implications": ["..."],
    "long_term_implications": ["..."]
  },
  "evolution_scenario": {
    "optimistic_scenario_description": "...",
    "optimistic_scenario_points": ["..."],
    "pessimistic_scenario_description": "...",
    "pessimistic_scenario_points": ["..."]
  },
  "topic_interconnexions": {
    "interconnexions": ["..."],
    "ripple_effects": ["..."]
  },
  "drivers_inhibitors": {
    "drivers": ["..."],
    "inhibitors": ["..."]
  }
}
""",
    "fr": """
1. Analyse de l'Impact Potentiel :
   - Examinez les effets potentiels de ce signal sur divers secteurs, industries et aspects sociétaux.
   - Considérez les implications à court et à long terme.
   - Analysez les effets d'entraînement possibles et les conséquences de second ordre.

2. Scénarios d'Évolution :
   - Décrivez les façons potentielles dont ce signal pourrait se développer ou se manifester à l'avenir.
   - Considérez divers facteurs qui pourraient influencer sa trajectoire.
   - Explorez des scénarios optimistes et pessimistes.

3. Interconnexions et Synergies :
   - Identifiez comment ce signal pourrait interagir avec d'autres tendances actuelles ou phénomènes émergents.
   - Discutez des synergies ou conflits potentiels avec les systèmes ou paradigmes existants.

4. Moteurs et Inhibiteurs :
   - Analysez les facteurs qui pourraient accélérer ou amplifier ce signal.
   - Examinez les obstacles ou résistances potentiels qui pourraient entraver son développement.

Votre analyse doit être approfondie et nuancée, allant au-delà des observations superficielles. Appuyez-vous sur votre expertise pour fournir des insights qui capturent la complexité et l'importance potentielle de ce signal. N'hésitez pas à faire des prédictions bien raisonnées sur sa trajectoire et son impact potentiels.

Concentrez-vous sur la fourniture d'une analyse claire, perspicace et exploitable qui peut éclairer la prise de décision stratégique et la planification future.

Retournez UNIQUEMENT un JSON valide utilisant les clés canoniques suivantes (les clés doivent rester en anglais, les valeurs peuvent être en français). Si une information manque, retournez des listes vides pour ces champs. N'ajoutez aucun champ supplémentaire ni de texte hors JSON.

{
  "potential_implications": {
    "short_term_implications": ["..."],
    "long_term_implications": ["..."]
  },
  "evolution_scenario": {
    "optimistic_scenario_description": "...",
    "optimistic_scenario_points": ["..."],
    "pessimistic_scenario_description": "...",
    "pessimistic_scenario_points": ["..."]
  },
  "topic_interconnexions": {
    "interconnexions": ["..."],
    "ripple_effects": ["..."]
  },
  "drivers_inhibitors": {
    "drivers": ["..."],
    "inhibitors": ["..."]
  }
}
""",
    "de": """
1. Potenzielle Auswirkungsanalyse:
   - Untersuchen Sie die potenziellen Auswirkungen dieses Signals auf verschiedene Sektoren, Branchen und gesellschaftliche Aspekte.
   - Berücksichtigen Sie sowohl kurzfristige als auch langfristige Implikationen.
   - Analysieren Sie mögliche Kaskadeneffekte und Folgen zweiter Ordnung.

2. Entwicklungsszenarien:
   - Beschreiben Sie potenzielle Wege, wie sich dieses Signal entwickeln oder in der Zukunft manifestieren könnte.
   - Berücksichtigen Sie verschiedene Faktoren, die seine Trajektorie beeinflussen könnten.
   - Erkunden Sie sowohl optimistische als auch pessimistische Szenarien.

3. Interkonnexionen und Synergien:
   - Identifizieren Sie, wie dieses Signal mit anderen aktuellen Trends oder aufkommenden Phänomenen interagieren könnte.
   - Diskutieren Sie potenzielle Synergien oder Konflikte mit bestehenden Systemen oder Paradigmen.

4. Treiber und Inhibitoren:
   - Analysieren Sie Faktoren, die dieses Signal beschleunigen oder verstärken könnten.
   - Untersuchen Sie potenzielle Barrieren oder Widerstände, die seine Entwicklung behindern könnten.

Ihre Analyse sollte gründlich und nuanciert sein und über oberflächliche Beobachtungen hinausgehen. Stützen Sie sich auf Ihre Expertise, um Erkenntnisse zu liefern, die die Komplexität und potenzielle Bedeutung dieses Signals erfassen. Zögern Sie nicht, gut begründete Vorhersagen über seine potenzielle Trajektorie und Auswirkungen zu machen.

Konzentrieren Sie sich auf die Bereitstellung einer klaren, einsichtsvollen und umsetzbaren Analyse, die strategische Entscheidungsfindung und zukünftige Planung informieren kann.

Geben Sie NUR gültiges JSON zurück und verwenden Sie die folgenden kanonischen Schlüssel (Schlüssel bleiben auf Englisch, Werte können auf Deutsch sein). Falls Informationen fehlen, geben Sie leere Listen für diese Felder zurück. Fügen Sie keine zusätzlichen Felder oder Text außerhalb des JSON hinzu.

{
  "potential_implications": {
    "short_term_implications": ["..."],
    "long_term_implications": ["..."]
  },
  "evolution_scenario": {
    "optimistic_scenario_description": "...",
    "optimistic_scenario_points": ["..."],
    "pessimistic_scenario_description": "...",
    "pessimistic_scenario_points": ["..."]
  },
  "topic_interconnexions": {
    "interconnexions": ["..."],
    "ripple_effects": ["..."]
  },
  "drivers_inhibitors": {
    "drivers": ["..."],
    "inhibitors": ["..."]
  }
}
""",
}

TOPIC_SUMMARY_PROMPT = {
    "en": """
As an expert analyst specializing in trend analysis and strategic foresight, your task is to provide a comprehensive evolution summary of Topic {topic_number}. Use only the information provided below:

{content_summary}

Structure your analysis as follows:

For the first timestamp:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame - format %Y-%m-%d]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

For all subsequent timestamps:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame - format %Y-%m-%d]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

### What's New
[1-2 sentences highlighting how this period differs from the previous one, focusing on new elements or significant changes]

Provide your analysis using only this format, based solely on the information given. Do not include any additional summary or overview sections beyond what is specified in this structure.
""",
    "fr": """
En tant qu'analyste expert spécialisé dans l'analyse des tendances et la prospective stratégique, votre tâche est de fournir un résumé complet de l'évolution du Sujet {topic_number}. Utilisez uniquement les informations fournies ci-dessous :

{content_summary}

Structurez votre analyse comme suit :

Pour le premier timestamp :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente - format %Y-%m-%d]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

Pour tous les timestamps suivants :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente - format %Y-%m-%d]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

### Nouveautés
[1-2 phrases soulignant en quoi cette période diffère de la précédente, en se concentrant sur les nouveaux éléments ou les changements significatifs]

Fournissez votre analyse en utilisant uniquement ce format, basé uniquement sur les informations données. N'incluez pas de sections de résumé ou d'aperçu supplémentaires au-delà de ce qui est spécifié dans cette structure.
""",
    "de": """
Als Expertenanalyst, der sich auf Trendanalyse und strategische Prognose spezialisiert hat, ist es Ihre Aufgabe, eine umfassende Entwicklungszusammenfassung von Thema {topic_number} zu erstellen. Verwenden Sie nur die unten angegebenen Informationen:

{content_summary}

Strukturieren Sie Ihre Analyse wie folgt:

Für den ersten Zeitstempel:

## [Prägnanter aber wirkungsvoller Titel, der das Wesen des Themas zu diesem Zeitpunkt erfasst]
### Datum: [Relevantes Datum oder Zeitrahmen - Format %Y-%m-%d]
### Wichtige Entwicklungen
- [Aufzählungspunkt, der eine wichtige Entwicklung oder einen Trend zusammenfasst]
- [Zusätzliche Aufzählungspunkte nach Bedarf]

### Analyse
[2-3 Sätze, die tiefere Einblicke in die Entwicklungen, ihre potenziellen Auswirkungen und ihre Bedeutung im breiteren Kontext der Entwicklung des Themas bieten]

Für alle nachfolgenden Zeitstempel:

## [Prägnanter aber wirkungsvoller Titel, der das Wesen des Themas zu diesem Zeitpunkt erfasst]
### Datum: [Relevantes Datum oder Zeitrahmen - Format %Y-%m-%d]
### Wichtige Entwicklungen
- [Aufzählungspunkt, der eine wichtige Entwicklung oder einen Trend zusammenfasst]
- [Zusätzliche Aufzählungspunkte nach Bedarf]

### Analyse
[2-3 Sätze, die tiefere Einblicke in die Entwicklungen, ihre potenziellen Auswirkungen und ihre Bedeutung im breiteren Kontext der Entwicklung des Themas bieten]

### Was ist neu
[1-2 Sätze, die hervorheben, wie sich diese Periode von der vorherigen unterscheidet, mit Fokus auf neue Elemente oder signifikante Änderungen]

Geben Sie Ihre Analyse nur in diesem Format basierend ausschließlich auf den gegebenen Informationen. Fügen Sie keine zusätzlichen Zusammenfassungs- oder Übersichtsabschnitte über das hinaus hinzu, was in dieser Struktur spezifiziert ist.
""",
}


def get_prompt(
    language: str,
    prompt_type: str,
    topic_number: int = None,
    content_summary: str = None,
    summary_from_first_prompt: str = None,
):
    # Map language to prompt language code
    if language == "English":
        lang = "en"
    elif language == "French":
        lang = "fr"
    elif language == "German":
        lang = "de"
    else:
        # Default to German for multilingual or unknown languages
        lang = "de"

    if prompt_type == "weak_signal":
        prompt = (
            SIGNAL_INTRO[lang].format(
                summary_from_first_prompt=summary_from_first_prompt
            )
            + SIGNAL_INSTRUCTIONS[lang]
        )

    elif prompt_type == "topic_summary":
        prompt = TOPIC_SUMMARY_PROMPT[lang].format(
            topic_number=topic_number, content_summary=content_summary
        )

    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    return prompt


def save_html_output(html_output, output_file="signal_llm.html"):
    """Function to save the model's output as HTML"""
    output_path = OUTPUT_PATH / output_file

    # Save the cleaned HTML
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_output)
    logger.debug(f"Cleaned HTML output saved to {output_path}")


def fill_html_template(
    topic_summary_list: TopicSummaryList,
    signal_analysis: SignalAnalysis,
    language: str = "de",
) -> str:
    """Fill the HTML template with appropriate data"""
    # Setup Jinja2 environment
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
    )
    
    # Select template based on language
    if language == "en":
        template_name = "signal_llm_template_en.html"
    elif language == "fr":
        template_name = "signal_llm_template_fr.html"
    elif language == "de":
        template_name = "signal_llm_template_de.html"
    else:
        # Default to German template for multilingual or unknown languages
        template_name = "signal_llm_template_de.html"
    
    template = env.get_template(template_name)

    # Sort the list by date from most recent to least recent
    try:
        sorted_topic_summary_by_time_period = sorted(
            topic_summary_list.topic_summary_by_time_period,
            key=lambda x: datetime.strptime(x.date, "%Y-%m-%d"),
            reverse=True,
        )
        topic_summary_list.topic_summary_by_time_period = (
            sorted_topic_summary_by_time_period
        )
    except Exception as e:
        logger.warning("Cannot sort summaries by date, probably wrong date format")

    # Render the template with the provided data
    rendered_html = template.render(
        topic_summary_list=topic_summary_list, signal_analysis=signal_analysis
    )

    # FIXME: many \n are added...
    rendered_html = rendered_html.replace("\n", "")
    rendered_html = rendered_html.replace("\\'", "'")

    return rendered_html
