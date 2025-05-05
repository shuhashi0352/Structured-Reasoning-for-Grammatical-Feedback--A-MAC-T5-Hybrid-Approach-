'''
Text normalisation functions

Copyright: S&I Challenge 2024
'''
import re
import os
import copy
import json
import string
import inflect
import logging
import unidecode
import argparse
from num2words import num2words
pinf = inflect.engine()

def lower_case(text):
    text = text.lower()
    return text

def remove_punctuation(text):
    text = text.strip("!,.:;?")
    return text

def remove_hesitation(text): 
    heslist=["aaa","ach","ah","ahh","ahm","aww","eee","eh","ehh","ehm","er","erm","err","ew",
         "ha","hee","hm","hmm","hmmm","huh","huhuh","mhm","mm","mmhm","mmhmm","mmm","mmmm","mmmmm","mmmmmm",
         "neh","uaaaa","uh","uhh","uhhh","uhhhh","uhhhhh","uhm","uhuh","uhhuh","um","umm","ummm","ummmmm",
         "huh-uh","uh-uh","mm-hm","mm-hmm","mm-huh","uh-hmm","uh-huh","um-hmm","um-hum"]
    if text.lower() in heslist:
        text = ""
    else:
        text = text
    return text

def ordinal_numbers(text):
    ordinal_matches = re.findall(r"(\d+(st|nd|rd|th))", text)
    for full_match, suffix in ordinal_matches:
        num = int(full_match[:-len(suffix)])
        text = text.replace(full_match, num2words(num, ordinal=True).lower())

    return text

def num2word(text):
    # Converts numbers (integers, floats, and large numbers with commas) into words
    words = text.split()

    for i, word in enumerate(words):
        clean_word = word.replace(",", "")
        
        if clean_word.replace(".", "").isdigit():  
            try:
                if "." in clean_word:
                    num = float(clean_word)
                else:
                    num = int(clean_word)
                
                words[i] = pinf.number_to_words(num)
            except ValueError:
                pass  # Ignore words that cannot be converted
    
    return " ".join(words)

def convert_to_ascii(text):
    text = unidecode.unidecode(text)
    return text

def remove_hyphens(text): # Preserve hyphenated words with letters and numbers intact.
    # Split hyphenated words with only letters into separate words. (e.g., "text-to-speech' -> 'text to speech")
    while re.search(r'([a-zA-Z]+)-([a-zA-Z]+)', text):
        text = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', text)
    # Split hyphenated words with only numbers into separate numbers. (e.g., "24-7" -> "24 7")
    while re.search(r'(\d+)-(\d+)', text):
        text = re.sub(r'(\d+)-(\d+)', r'\1 \2', text)
    return text.strip()

def remove_partial_words(text): # remove partial words like 'wo-' -> ''
    text = re.sub(r'\b[a-zA-Z]+-(?!\d|[a-zA-Z])', '', text)
    return text

def split_token(text):
    # split currency token from item and reorder
    text = re.sub(r'([$£€¥])(\d+)', r'\2 \1', text) # Reorder '$100' -> '100 $', etc.
    # split percent token from item
    text = re.sub(r'(\d+)([%°])', r'\1 \2', text)   # Reorder '100%' -> '100 %', etc.
    # split tokens with & between them e.g. 'r&b' -> 'r and b'
    text = re.sub(r'(\w+)&(\w+)', r'\1 and \2', text)
    return text.strip()

# Special symbols mapping
special_symbols_dict = {
    # Currency Symbols
    "$": "dollars",
    "£": "pounds",
    "€": "euros",
    "¥": "yen",

    # Measurement Symbols
    "%": "percent",
    "°c": "degrees celsius",
    "°": "degrees",

    # Common Symbols
    "&": "and",
    "@": "at",

    # Other Symbols
    "§": "",
    "©": "",
    "®": "",
    "™": "",
}

def special_symbols(text):
    text = split_token(text)
    words = text.split()  # Split the input text into words
    for i, word in enumerate(words):
        if word in special_symbols_dict:
            words[i] = special_symbols_dict[word]  # Replace with mapped value
    return ' '.join(words) 




