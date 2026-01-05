import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. SETUP: Define the Output Structure (The Prescription) ---
# We use Pydantic to force the AI to return structured JSON data, not just text.
class MusicPrescription(BaseModel):
    diagnosis: str = Field(description="A brief, empathetic analysis of the user's emotional state.")
    mode_name: str = Field(description="The name of the Dastgah or Avaz prescribed (e.g., 'Avaz-e Dashti').")
    archetype: str = Field(description="The archetype of this mode (e.g., 'The Shepherd', 'The Warrior', 'The Hacker').")
    neuroscience_mechanism: str = Field(description="The scientific explanation of why this works (e.g., Dopamine, Prolactin, Beta waves).")
    search_query: str = Field(description="A specific search string for YouTube/Spotify (e.g., 'Kayhan Kalhor Dashti Improvisation').")
    activity_suggestion: str = Field(description="What the user should do while listening.")

# --- 2. CONFIGURATION: The Pharmacist's Brain ---
# This is where we teach the AI the "Rules" of Persian Music Therapy.

PHARMACOPEIA_CONTEXT = """
You are 'Dr. Radif', a highly sophisticated Music Therapist specializing in the neuroscience of Persian Traditional Music. 
Your goal is to analyze the user's context/mood and prescribe the exact Persian musical mode (Dastgah or Avaz) to regulate their nervous system.

Use this internal Pharmacopeia (Knowledge Base) to make your decision:

1. **CONTEXT: Grief, Depression, Broken Heart, Rain**
   - **RX:** Avaz-e Dashti (The Shepherd)
   - **Mechanism:** Mimics human crying frequencies; triggers Prolactin release for 'The Good Cry'.

2. **CONTEXT: Morning, Low Energy, Need Motivation, Sunrise**
   - **RX:** Dastgah-e Mahur (The Sunrise) or Chahargah (The Warrior)
   - **Mechanism:** Mahur triggers Dopamine (joy). Chahargah triggers Adrenaline (power/fight-or-flight).

3. **CONTEXT: Nostalgia, Missing Home, Sunset, Autumn, Romantic Longing**
   - **RX:** Avaz-e Bayat-e Isfahan (The Mystic) or Segah
   - **Mechanism:** Uses microtones that stimulate the Hippocampus (memory) and Default Mode Network. Creates 'Saudade'.

4. **CONTEXT: Anxiety, Panic, Chaos, Overwhelmed**
   - **RX:** Dastgah-e Homayoun (The King)
   - **Mechanism:** Majestic and gravity-heavy. Forces brainwaves from erratic Beta to focused Alpha. Grounding.

5. **CONTEXT: Deep Focus, Coding, Math, Logic Work**
   - **RX:** Dastgah-e Nava (The Mathematician) or Rast-Panjgah
   - **Mechanism:** Complex, repetitive, chant-like structures. Stabilizes High-Beta waves. Creates 'Flow State'.

6. **CONTEXT: Insomnia, Bedtime, Racing Thoughts**
   - **RX:** Avaz-e Abu Ata (The Philosopher)
   - **Mechanism:** Low-frequency bias, no sudden climaxes. Induces Delta waves.

7. **CONTEXT: Vacation, Nature, Reset, Decompression**
   - **RX:** Bayat-e Tork (The Traveler) or Afshari
   - **Mechanism:** Linked to Sufi storytelling and open horizons. Breaks the urban 'grid'.

**INSTRUCTIONS:**
- Analyze the user's input for emotional nuance (e.g., 'bittersweet' is Isfahan, 'crushed' is Dashti).
- Return the output strictly in the requested JSON format.
"""

def get_persian_music_prescription(user_input: str, openai_api_key: str):
    """
    Takes a natural language input and returns a structured music prescription.
    """
    
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0.7, 
        model="gpt-4-turbo", 
        openai_api_key=openai_api_key
    )

    # Set up the Parser
    parser = PydanticOutputParser(pydantic_object=MusicPrescription)

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(PHARMACOPEIA_CONTEXT),
        HumanMessagePromptTemplate.from_template(
            "PATIENT SYMPTOMS: {input}\n\n"
            "Analyze the symptoms and provide the prescription.\n"
            "{format_instructions}"
        )
    ])

    # Build the Chain
    chain = prompt | llm | parser

    # Run the Chain
    try:
        result = chain.invoke({
            "input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        return f"Error: {e}"

# --- 3. EXECUTION: The User Interface (Simulated) ---

if __name__ == "__main__":
    # ⚠️ REPLACE WITH YOUR ACTUAL KEY or set as env variable
    my_api_key = os.getenv("OPENAI_API_KEY") 
    
    if not my_api_key:
        print("Please set your OPENAI_API_KEY environment variable.")
    else:
        print("--- DR. RADIF IS LISTENING ---")
        
        # Scenarios to test the dynamic logic
        user_scenarios = [
            "I have to write 500 lines of Python code and I'm distracted.",
            "I just went through a breakup and it's raining outside.",
            "I'm feeling super anxious about a meeting and my heart is racing.",
            "I'm on a road trip in the desert and feeling philosophical."
        ]

        for symptom in user_scenarios:
            print(f"\n🗣️ PATIENT: '{symptom}'")
            rx = get_persian_music_prescription(symptom, my_api_key)
            
            if isinstance(rx, MusicPrescription):
                print(f"💊 RX: {rx.mode_name} ({rx.archetype})")
                print(f"🧠 WHY: {rx.neuroscience_mechanism}")
                print(f"🔍 SEARCH: '{rx.search_query}'")
                print(f"💡 DO THIS: {rx.activity_suggestion}")
                print("-" * 50)
