# import google.generativeai as genai
# import os

# # Apni actual API key yahan daalo
# genai.configure(api_key="AQ.Ab8RN6Iy64sJjwBIA9C-qzW2G2CjtSMyHujQG2jtovRLhfdFLQ")

# print("Available Models for Text Generation:")
# print("-" * 40)

# # Ye loop sabhi models ko fetch karega
# for m in genai.list_models():
#     # Hum sirf wo models filter kar rahe hain jo text generate kar sakte hain
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)









# import requests

# # Apni Google API Key yahan dalo
# GOOGLE_API_KEY = "AQ.Ab8RN6Iy64sJjwBIA9C-qzW2G2CjtSMyHujQG2jtovRLhfdFLQ"

# def check_embedding_models():
#     print("Checking live models from Google Server...\n")
#     url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
    
#     try:
#         response = requests.get(url)
        
#         if response.status_code != 200:
#             print("❌ Error connecting to Google API:")
#             print(response.json())
#             return
            
#         models = response.json().get("models", [])
        
#         print("✅ Available Embedding Models:\n")
#         found = False
#         for model in models:
#             # Sirf wo models check karo jo "embedContent" (Embeddings) support karte hain
#             if "embedContent" in model.get("supportedGenerationMethods", []):
#                 print(f"👉 Exact Model Name: '{model['name']}'")
#                 found = True
                
#         if not found:
#             print("Koi embedding model nahi mila. API key check karo.")
            
#     except Exception as e:
#         print(f"Network Error: {e}")

# if __name__ == "__main__":
#     check_embedding_models()