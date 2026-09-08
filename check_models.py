import google.generativeai as genai
import os

# Apni actual API key yahan daalo
genai.configure(api_key="AQ.Ab8RN6Iy64sJjwBIA9C-qzW2G2CjtSMyHujQG2jtovRLhfdFLQ")

print("Available Models for Text Generation:")
print("-" * 40)

# Ye loop sabhi models ko fetch karega
for m in genai.list_models():
    # Hum sirf wo models filter kar rahe hain jo text generate kar sakte hain
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)