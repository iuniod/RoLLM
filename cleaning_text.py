from typing import List, Dict, Match
import itertools
import re
from tqdm import tqdm
import emoji
import pandas as pd
import ftfy
import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    ''' Dataset class for text data '''
    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class BaseFormatter:
    ''' Base class for text formatters '''
    def __init__(self) -> None:
        pass

    def format(self, text: str) -> str:
        ''' Method to format text, to be implemented by subclasses '''
        raise NotImplementedError


class RomanianTextNormalization(BaseFormatter):
    name = "Romanian Diacritics / Text Normalization Formatter"

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _normalize_spacing_for_tok(text: str) -> str:
        res = (
            text.replace("\r", "")
            .replace("(", " (")
            .replace(")", ") ")
            .replace(" +", " ")
        )
        res = re.sub(r"\) ([\.\!\:\?\;\,])", r"\)\1", res)
        res = res.replace("( ", "(").replace(" )", ")")
        res = re.sub(r"(\d) \%", r"\1\%", res)
        res = res.replace(" :", ":").replace(" ;", ";")
        res = res.replace("`", "'").replace("''", ' " ')

        res = (
            res.replace("„", '"')
            .replace(""", '"')
            .replace(""", '"')
            .replace("–", "-")
            .replace("—", " - ")
            .replace(" +", " ")
            .replace("´", "'")
            .replace("([a-z])'([a-z])", r"\1'\2/")
            .replace("'", '"')
            .replace("‚", '"')
            .replace("''", '"')
            .replace("´´", '"')
            .replace("…", "...")
            .replace(" « ", ' "')
            .replace("« ", '"')
            .replace("«", '"')
            .replace(" » ", '" ')
            .replace(" »", '"')
            .replace("»", '"')
            .replace(" %", "%")
            .replace(" :", ":")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ;", ";")
            .replace(", ", ", ")
            .replace("．", ". ")
        )

        # Handle quotation marks
        res = re.sub(r"\"([,\.]+)", r"\1\"", res)

        # Handle number formatting
        res = re.sub(r"(\d) (\d)", r"\1 \2", res)

        # Convert decimal points to commas for Romanian
        res = re.sub(r"(\d)\.(\d)", r"\1,\2", res)

        # Convert English-style quotes to Romanian-style „ ”
        res = res.replace('"', '„', 1).replace('"', '”', 1)

        return res

    def format(self, text: str) -> str:
        text = text.replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')
        text = self._normalize_spacing_for_tok(text)
        return text


class LineFormatter(BaseFormatter):
    name = "Romanian Line Formatter"

    def __init__(self) -> None:
        super().__init__()

    def format(self, text: str) -> str:
        lines = text.split('\n')
        final_lines = []

        for line in lines:
            words = ftfy.fix_text(line).split()

            if len(words) <= 1:
                continue

            if line.isupper() or line.isdigit():
                continue

            num_digits = sum(c.isdigit() for c in line)

            if num_digits / len(line) > 0.5:
                continue

            num_caps = sum(c.isupper() for c in line)

            if num_caps / len(line) > 0.75:
                continue

            if line.count(' | ') >= 1:
                continue

            final_lines.append(line)

        return '\n'.join(final_lines)


class RegexFormatter(BaseFormatter):
    name = "Regex Formatter (URLs, Emojis, phone numbers, etc)"

    def __init__(self) -> None:
        super().__init__()
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.[\w-]+[\.[\w.-]+]+'
        )
        self.phone_pattern = re.compile(r'(\+40\s?|0)\d{2}[\s.-]?\d{3}[\s.-]?\d{3}')
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.date_patterns = [
            (re.compile(r'(\d{4})-(\d{2})-(\d{2})'), r'\3.\2.\1'),  # YYYY-MM-DD → DD.MM.YYYY
            (re.compile(r'(\d{2})/(\d{2})/(\d{4})'), r'\1.\2.\3')   # MM/DD/YYYY → DD.MM.YYYY
        ]
        self.romanian_date_pattern = re.compile(r'(\d{1,2}) (\w+) (\d{4})')

    @staticmethod
    def _generate_month_dict() -> Dict[str, str]:
        ''' Generates a dictionary for month names '''
        base_months = {
            "ianuarie": "01", "februarie": "02", "martie": "03", "aprilie": "04",
            "mai": "05", "iunie": "06", "iulie": "07", "august": "08",
            "septembrie": "09", "octombrie": "10", "noiembrie": "11", "decembrie": "12",
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07",
            "september": "09", "october": "10", "november": "11", "december": "12"
        }

        # Add short forms and capitalize versions dynamically
        return {month.lower(): num for month, num in base_months.items()} | \
               {month[:3].lower(): num for month, num in base_months.items()}

    def _replace_romanian_date(self, match: Match[str]) -> str:
        ''' Converts Romanian date format to DD.MM.YYYY '''
        day, month, year = match.groups()
        month_number = self._generate_month_dict().get(month.lower())

        return f"{day.zfill(2)}.{month_number}.{year}" if month_number else match.group(0)


    def format(self, text: str) -> str:
        text = self.url_pattern.sub('www.exemplu.ro', text)
        text = self.phone_pattern.sub('07XXXXXXXX', text)
        text = self.email_pattern.sub('email@exemplu.com', text)

        for pattern, replacement in self.date_patterns:
            text = pattern.sub(replacement, text)

        # Normalize Romanian-style dates
        text = self.romanian_date_pattern.sub(self._replace_romanian_date, text)

        # Replace emojis and normalize spaces
        return re.sub(r'\s+', ' ', emoji.replace_emoji(text)).strip()


if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_parquet("./datasets/ro_part_00000.parquet")
    texts = df['text'].to_list()

    # Create Dataset & DataLoader
    dataset = TextDataset(texts)
    BATCH_SIZE = 1024
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Process text
    formatters = [
        RomanianTextNormalization(),
        RegexFormatter(),
        LineFormatter()
    ]

    for formatter in formatters:
        texts = list(itertools.chain.from_iterable(
            [formatter.format(text) for text in batch] for
                batch in tqdm(dataloader, desc=f"Processing: {formatter.__class__.__name__}")
        ))

    df['text'] = texts

    df.to_parquet("./datasets/ro_part_00000_cleaned.parquet", compression="snappy")

    print("Processing complete! ✅")
