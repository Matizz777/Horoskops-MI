from transformers import GPT2LMHeadModel, GPT2Tokenizer
from skyfield.api import load
from datetime import datetime
import random

class AstroloģiskaisAIHoroskops:
    def __init__(self):
        # Load the trained GPT-2 model and tokenizer
        self.model_name = "./trained_model"  # Path to the trained model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # Set the pad token if not already set
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load planetary data from Skyfield
        self.ephemeris = load('de421.bsp')
        self.zeme = self.ephemeris['earth']
        self.planetas = [
            "jupiter barycenter",
            "saturn barycenter",
            "uranus barycenter",
            "neptune barycenter",
            "pluto barycenter",
            "mercury",
            "venus",
            "mars",
            "sun"
        ]

        # Zodiac signs
        self.zodiaka_zimes = [
            "Auns", "Vērsis", "Dvīņi", "Vēzis", "Lauva", "Jaunava",
            "Svari", "Skorpions", "Strēlnieks", "Mežāzis", "Ūdensvīrs", "Zivis"
        ]

    def iegut_planetu_pozicijas(self):
        """Calculate the planetary positions relative to Earth for the current date."""
        t = load.timescale().now()
        pozicijas = {}
        for planeta in self.planetas:
            planeta_obj = self.ephemeris[planeta]
            pozicija = self.zeme.at(t).observe(planeta_obj).apparent().ecliptic_latlon()
            pozicijas[planeta] = pozicija[0].degrees
        return pozicijas

    def izgeneret_horoskopu(self, zodiaka_zime, pozicijas):
        """Ģenerē horoskopu un cilvēka īpašību aprakstu latviski."""
        # Format planetary descriptions
        planētu_apraksts = ", ".join([f"{planeta}: {round(pozicija, 2)}°" for planeta, pozicija in pozicijas.items()])
        
        # Create a prompt in Latvian
        prompt = (
            f"Planētu novietojums šodien ({planētu_apraksts}) sniedz šādu ieskatu "
            f"par {zodiaka_zime} cilvēku: "
            "Apraksti viņa galvenās rakstura īpašības, ikdienas emocijas un planētu ietekmi uz darbu un attiecībām."
        )

        # Prepare inputs for the model
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate text with attention_mask and pad_token_id
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=250,
            num_return_sequences=1,
            no_repeat_ngram_size=3,  # Palieliniet no_repeat_ngram_size
            temperature=0.9,         # Palieliniet randomness
            top_p=0.85,              # Samaziniet top-p (mazāks fokuss uz visbiežākajiem vārdiem)
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode the generated text
        horoskops = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return horoskops
    
    def masveida_generācija(self, skaits=10):
        """Masveidā ģenerē horoskopus visām zodiaka zīmēm."""
        horoskopi = []  # Initialize an empty list to store horoscopes
        for _ in range(skaits):
            pozicijas = self.iegut_planetu_pozicijas()  # Calculate positions for each iteration
            zodiaka_zime = random.choice(self.zodiaka_zimes)  # Randomly select a zodiac sign
            horoskops = self.izgeneret_horoskopu(zodiaka_zime, pozicijas)  # Generate horoscope
            horoskopi.append(horoskops)  # Append the generated horoscope to the list
        return horoskopi  # Return the list of horoscopes

# Create an instance of the generator
generators = AstroloģiskaisAIHoroskops()

# Generate horoscopes and save them to a file
horoskopi = generators.masveida_generācija(10)  # Generate 10 horoscopes for testing
with open("ai_horoskopi.txt", "w", encoding="utf-8") as fails:
    fails.write("\n\n".join(horoskopi))  # Separate each horoscope with a blank line

print("Horoskopu ģenerēšana pabeigta!")