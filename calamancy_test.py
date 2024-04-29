import calamancy

# Load the Tagalog language model
nlp = calamancy.load("tl_calamancy_lg-0.1.0")

# Your Tagalog text for NER
text = "Ako si Christian, ako ay nag tatrabaho sa Red Cross at nakatira ako sa Davao"

# Process the text with the loaded model
doc = nlp(text)

# Print detected entities
for ent in doc.ents:
    print(ent.text, ent.label_)
