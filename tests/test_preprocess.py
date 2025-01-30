import unittest
from src.preprocess import preprocess_text

class TestPreprocess(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a TEST sentence!"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "this is a test sentence")

if __name__ == '__main__':
    unittest.main()
