import os
from pathlib import Path
from dotenv import load_dotenv

# ✅ Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

print("\n🔍 Checking Gemini API Key...\n")

if not api_key:
    print("❌ GEMINI_API_KEY NOT FOUND")
    exit()
else:
    print("✅ GEMINI_API_KEY FOUND")

try:
    import google.generativeai as genai

    # ✅ Configure API
    genai.configure(api_key=api_key)

    print("\n📡 Fetching available models...\n")

    models = genai.list_models()

    supported_models = []

    for m in models:
        if "generateContent" in m.supported_generation_methods:
            supported_models.append(m.name)

    print("✅ Supported Models:\n")
    for model in supported_models:
        print("👉", model)

    # ---------------------------------------------------
    # 🔥 TEST ACTUAL GEMINI CALL (IMPORTANT)
    # ---------------------------------------------------

    print("\n🚀 Testing Gemini call with gemini-2.5-flash...\n")

    try:
        model_name = "gemini-2.5-flash"   # ✅ correct format

        model = genai.GenerativeModel(model_name)

        response = model.generate_content("Say hello in one short sentence.")

        print("✅ Gemini call SUCCESS\n")
        print("Response:", response.text)

    except Exception as e:
        print("\n🔥 GEMINI CALL FAILED 🔥")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

        # Extra debug
        if hasattr(e, "response"):
            print("\nFull response:", e.response)

except Exception as e:
    print("\n❌ ERROR while fetching models:")
    print(type(e).__name__, str(e))