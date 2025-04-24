"""Utility functions for text cleaning and rewriting.from_rag_to_riches"""

import re
from openai import OpenAI
from rag_project.config import settings

def clean_text(text):
    """
    Clean text by removing markdown links, URLs, HTML tags, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'http\S+', '', text)         # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)         # Remove HTML tags
    text = re.sub(r'© .*?\n?', '', text)        # Remove photo credits
    text = re.sub(r'\s+', ' ', text)            # Reduce spaces
    text = re.sub(r'(Mis à jour le.*?\n|Suivez nous !|Lien affilié Amazon|Crédit.*?\n)', '', text, flags=re.IGNORECASE)
    return text.strip()

def rewrite_text(text, model="gpt-4.1-nano", temperature=0.7, max_tokens=3000):
    """
    Rewrite text using OpenAI's API for a cleaner, more informative version.
    
    Args:
        text (str): Text to rewrite
        model (str): OpenAI model to use
        temperature (float): Temperature setting for text generation
        max_tokens (int): Maximum tokens for the generated text
        
    Returns:
        str: Rewritten text
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    prompt = f"""
    Tu es un rédacteur web SEO. Réécris le texte ci-dessous avec une formulation différente, tout en gardant les mots-clés importants (comme nutrition, compléments alimentaires, vitamines, performance, musculation, etc.). Supprime les contenus promotionnels, les liens, les mentions légales ou cookies. Le texte doit rester informatif, clair, et unique.
    
    Texte :
    \"\"\"
    {text}
    \"\"\"
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content