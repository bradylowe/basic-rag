from dotenv import load_dotenv
import os


load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
DATA_PATH = os.getenv('DATA_PATH', 'data')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
