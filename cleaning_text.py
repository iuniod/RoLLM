import pandas as pd
import re
from langdetect import detect, LangDetectException
from tqdm import tqdm

def clean_romanian_dataset(texts, min_length=10):
	cleaned_texts = []

	for text in tqdm(texts, desc='Cleaning texts'):
		# Step 1: Remove extra whitespace and newlines
		# text = re.sub(r'\s+', ' ', text).strip()

		# Step 2: Filter out non-Romanian text
		try:
			if detect(text) != 'ro':
				continue
		except LangDetectException:
			continue

		# Step 3: Normalize diacritics - ensure consistent use of them
		text = text.replace('ş', 'ș').replace('ţ', 'ț')
		text = text.replace('Ş', 'Ș').replace('Ţ', 'Ț')

		# Step 4: Remove special characters and numbers
		text = re.sub(r"[^a-zA-ZăâîșțĂÂÎȘȚ0-9\s.,;:!?()\[\]{}\"'\-—_/\\@#&*%+=<>]", "", text)

		# Step 5: Convert to lowercase
		# text = text.lower()

		# Step 6: Filter out short texts
		if len(text.split()) < min_length:
			continue

		cleaned_texts.append(text)

	return cleaned_texts


if __name__ == "__main__":
	# Load your dataset
	df = pd.read_parquet("./datasets/ro_part_00000.parquet")
	print("Number of texts before cleaning: ", len(df["text"]))

	cleaned_texts = clean_romanian_dataset(df["text"])
	print("Number of texts after cleaning: ", len(cleaned_texts))

	# Save the cleaned text to a file
	with open("./datasets/ro_part_00000.parquet_texts_cleaned.txt", "w") as f:
		for line in cleaned_texts:
			f.write(line + "\n")

	chars = sorted(list(set(''.join(cleaned_texts))))
	print(''.join(chars))