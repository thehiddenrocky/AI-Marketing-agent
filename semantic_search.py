import json
import glob
import os
from typing import List, Dict, Optional, Any, NamedTuple
import numpy as np
import re
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SeedSentence:
    """Data class to store seed sentence information"""
    text: str
    category: str


@dataclass
class SentenceData:
    """Data class to store sentence information"""
    sentence: str
    source: str
    post_id: str
    comment_id: Optional[str]
    author: str
    text: str
    permalink: str
    similarity_score: Optional[float] = None
    matched_seed: Optional[str] = None
    category: Optional[str] = None


class SemanticSearch:
    def __init__(self, model: str = "text-embedding-3-large"):
        """Initialize SemanticSearch with API configuration"""
        load_dotenv()

        # Load API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = 100

    def collect_sentences(self, data_pattern: str = '*_data.json') -> List[SentenceData]:
        """Collect sentences from posts and comments with metadata"""
        sentence_data = []
        data_files = list(Path('.').glob(data_pattern))

        if not data_files:
            logger.warning(f"No files found matching pattern: {data_pattern}")
            return []

        for file_path in tqdm(data_files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._process_posts(data.get('posts', []), sentence_data)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        logger.info(f"Total sentences collected: {len(sentence_data)}")
        return sentence_data

    def _process_posts(self, posts: List[Dict], sentence_data: List[SentenceData]) -> None:
        """Process posts and their comments to extract sentences"""
        for post in posts:
            post_id = post.get('id')
            post_author = post.get('author', '')
            post_permalink = f"https://www.reddit.com{post.get('permalink', '')}"

            # Process post title
            if title := post.get('title', '').strip():
                sentence_data.append(SentenceData(
                    sentence=title,
                    source='post_title',
                    post_id=post_id,
                    comment_id=None,
                    author=post_author,
                    text=title,
                    permalink=post_permalink
                ))

            # Process post content
            if selftext := post.get('selftext', ''):
                self._process_text(
                    selftext, 'post_selftext', post_id, None,
                    post_author, post_permalink, sentence_data
                )

            # Process comments
            for comment in post.get('comments', []):
                if body := comment.get('body', ''):
                    comment_id = comment.get('id')
                    comment_author = comment.get('author', '')
                    comment_permalink = f"https://www.reddit.com{comment.get('permalink', '')}"
                    self._process_text(
                        body, 'comment', post_id, comment_id,
                        comment_author, comment_permalink, sentence_data
                    )

    def _process_text(self, text: str, source: str, post_id: str, comment_id: Optional[str],
                      author: str, permalink: str, sentence_data: List[SentenceData]) -> None:
        """Process text content and extract sentences"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sent in sentences:
            if sent := sent.strip():
                sentence_data.append(SentenceData(
                    sentence=sent,
                    source=source,
                    post_id=post_id,
                    comment_id=comment_id,
                    author=author,
                    text=text,
                    permalink=permalink
                ))

    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a list of texts"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [None] * len(texts)

    @staticmethod
    def compute_similarity(embedding_a: List[float], embedding_b: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, sentence_data: List[SentenceData],
               seed_sentences: List[SeedSentence], threshold: float = 0.85) -> List[SentenceData]:
        """Perform semantic search using embeddings"""
        logger.info("Generating embeddings for seed sentences...")
        seed_texts = [seed.text for seed in seed_sentences]
        seed_embeddings = self.get_embeddings(seed_texts)
        valid_seeds = [(seed, emb) for seed, emb in zip(seed_sentences, seed_embeddings) if emb is not None]

        if not valid_seeds:
            logger.error("No valid embeddings for seed sentences")
            return []

        logger.info("Processing sentences in batches...")
        matched_sentences = []

        for i in tqdm(range(0, len(sentence_data), self.batch_size)):
            batch = sentence_data[i:i + self.batch_size]
            sentences_batch = [item.sentence for item in batch]

            embeddings = self.get_embeddings(sentences_batch)

            for item, embedding in zip(batch, embeddings):
                if embedding is None:
                    continue

                # Find the best matching seed sentence
                max_similarity = 0
                best_seed = None

                for seed, seed_embedding in valid_seeds:
                    similarity = self.compute_similarity(embedding, seed_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_seed = seed

                if max_similarity >= threshold and best_seed:
                    item.similarity_score = max_similarity
                    item.matched_seed = best_seed.text
                    item.category = best_seed.category
                    matched_sentences.append(item)

        logger.info(f"Number of sentences matched: {len(matched_sentences)}")
        return matched_sentences


def main():
    """Main execution function"""
    # Initialize semantic search
    semantic_search = SemanticSearch()

    # Define seed sentences with categories
    seed_sentences = [
        # Productivity & Time Management
        SeedSentence("I need an app to automatically prioritize my daily tasks.", "Productivity & Time Management"),
        SeedSentence("It would be great if I could sync all my calendars without duplicates.",
                     "Productivity & Time Management"),
        SeedSentence("I have problems focusing on work when distractions pile up.", "Productivity & Time Management"),
        SeedSentence("I need a tool that helps me track progress on long-term goals.",
                     "Productivity & Time Management"),
        SeedSentence("It would be amazing to have a system that blocks unnecessary notifications during deep work.",
                     "Productivity & Time Management"),
        SeedSentence("I have problems managing overlapping project deadlines effectively.",
                     "Productivity & Time Management"),
        SeedSentence("I need software to estimate how long tasks will actually take.",
                     "Productivity & Time Management"),

        # Operations & Workflow
        SeedSentence("I need a system that flags workflow bottlenecks in real time.", "Operations & Workflow"),
        SeedSentence("It would be great to automate invoice approvals to save time.", "Operations & Workflow"),
        SeedSentence("I have problems coordinating between suppliers and internal teams.", "Operations & Workflow"),
        SeedSentence("I need software that predicts inventory shortages before they happen.", "Operations & Workflow"),
        SeedSentence("It would be helpful if operational reports could be auto-generated from raw data.",
                     "Operations & Workflow"),
        SeedSentence("I have problems scaling my workflows without adding more staff.", "Operations & Workflow"),
        SeedSentence("I need an app that shows where delays are occurring in my processes.", "Operations & Workflow"),

        # Finance & Costs
        SeedSentence("I need a way to track every dollar spent across my business.", "Finance & Costs"),
        SeedSentence("It would be great to have software that simplifies cash flow forecasting.", "Finance & Costs"),
        SeedSentence("I have problems figuring out which subscriptions I’m overpaying for.", "Finance & Costs"),
        SeedSentence("I need a tool that automatically tracks and categorizes expenses by team.", "Finance & Costs"),
        SeedSentence("It would be amazing to have an app that breaks down tax obligations in real time.",
                     "Finance & Costs"),
        SeedSentence("I have problems managing budgets for multiple projects simultaneously.", "Finance & Costs"),
        SeedSentence("I need a financial dashboard to monitor real-time profitability.", "Finance & Costs"),

        # Marketing & Sales
        SeedSentence("I need a tool that helps me craft personalized marketing messages for each customer.",
                     "Marketing & Sales"),
        SeedSentence("It would be great to have a platform to automate and track sales outreach.", "Marketing & Sales"),
        SeedSentence("I have problems identifying which marketing channels deliver the best ROI.", "Marketing & Sales"),
        SeedSentence("I need an AI tool to generate optimized ads for different platforms.", "Marketing & Sales"),
        SeedSentence("It would be amazing to track customer preferences automatically for tailored offers.",
                     "Marketing & Sales"),
        SeedSentence("I have problems closing sales because I lack clear insights into customer needs.",
                     "Marketing & Sales"),
        SeedSentence("I need software to qualify leads without manual data entry.", "Marketing & Sales"),

        # Team Management & HR
        SeedSentence("I need an app to make scheduling shifts less of a nightmare.", "Team Management & HR"),
        SeedSentence("It would be great to simplify the onboarding process for new hires.", "Team Management & HR"),
        SeedSentence("I have problems keeping my remote team engaged and motivated.", "Team Management & HR"),
        SeedSentence("I need a tool to track employee productivity without being invasive.", "Team Management & HR"),
        SeedSentence("It would be amazing to have software that collects and analyzes employee feedback.",
                     "Team Management & HR"),
        SeedSentence("I have problems ensuring that performance reviews are fair and consistent.",
                     "Team Management & HR"),
        SeedSentence("I need a solution to reduce employee turnover through better engagement.",
                     "Team Management & HR"),

        # Customer Service & Support
        SeedSentence("I need a system that prioritizes customer support tickets by urgency.",
                     "Customer Service & Support"),
        SeedSentence("It would be great if I could automatically gather feedback after each support interaction.",
                     "Customer Service & Support"),
        SeedSentence("I have problems responding to customers fast enough during peak times.",
                     "Customer Service & Support"),
        SeedSentence("I need a chatbot that works in multiple languages for global customers.",
                     "Customer Service & Support"),
        SeedSentence("It would be helpful if there were a tool to consolidate customer complaints into one view.",
                     "Customer Service & Support"),
        SeedSentence("I have problems tracking recurring customer issues and finding patterns.",
                     "Customer Service & Support"),
        SeedSentence("I need a way to predict which customers are likely to churn and why.",
                     "Customer Service & Support"),

        # Data & Technology
        SeedSentence("I need an app to clean and organize messy data sets automatically.", "Data & Technology"),
        SeedSentence("It would be great to have a dashboard that integrates metrics from all tools in one view.",
                     "Data & Technology"),
        SeedSentence("I have problems ensuring data security across multiple platforms.", "Data & Technology"),
        SeedSentence("I need a system that makes predictive analytics easy for non-technical teams.",
                     "Data & Technology"),
        SeedSentence("It would be helpful if my software automatically updated without breaking integrations.",
                     "Data & Technology"),
        SeedSentence("I have problems identifying actionable insights from large amounts of data.",
                     "Data & Technology"),
        SeedSentence("I need a tool to train employees on new technologies faster and more effectively.",
                     "Data & Technology"),

        # Growth & Strategy
        SeedSentence("I need a platform to help me identify emerging market trends before competitors.",
                     "Growth & Strategy"),
        SeedSentence("It would be great if I could get detailed competitor analysis without hiring consultants.",
                     "Growth & Strategy"),
        SeedSentence("I have problems scaling my business without sacrificing quality.", "Growth & Strategy"),
        SeedSentence("I need a system that tracks long-term growth metrics and suggests improvements.",
                     "Growth & Strategy"),
        SeedSentence("It would be amazing to test new business models without large upfront investments.",
                     "Growth & Strategy"),
        SeedSentence("I have problems keeping my pricing strategy competitive without losing margins.",
                     "Growth & Strategy"),
        SeedSentence("I need a tool to make strategic planning more data-driven and collaborative.",
                     "Growth & Strategy"),

        # Communication & Collaboration
        SeedSentence("I need a system to reduce miscommunication between teams in different time zones.",
                     "Communication & Collaboration"),
        SeedSentence("It would be great to consolidate all communication channels into one app.",
                     "Communication & Collaboration"),
        SeedSentence("I have problems keeping everyone aligned on project timelines.", "Communication & Collaboration"),
        SeedSentence("I need a tool to document meeting notes and actions automatically.",
                     "Communication & Collaboration"),
        SeedSentence("It would be amazing to streamline file sharing with proper version control.",
                     "Communication & Collaboration"),
        SeedSentence("I have problems ensuring that knowledge sharing is consistent across teams.",
                     "Communication & Collaboration"),
        SeedSentence("I need software that tracks project responsibilities and avoids duplicate efforts.",
                     "Communication & Collaboration"),

        # Compliance & Legal
        SeedSentence("I need a platform that tracks changes in regulations relevant to my business.",
                     "Compliance & Legal"),
        SeedSentence("It would be great to simplify contract management with automated reminders.",
                     "Compliance & Legal"),
        SeedSentence("I have problems ensuring data privacy compliance for international customers.",
                     "Compliance & Legal"),
        SeedSentence("I need a system to prepare for audits without last-minute scrambling.", "Compliance & Legal"),
        SeedSentence("It would be amazing if compliance training could be gamified for my employees.",
                     "Compliance & Legal"),
        SeedSentence("I have problems staying on top of legal documentation updates.", "Compliance & Legal"),
        SeedSentence("I need a dashboard that visualizes all compliance risks at a glance.", "Compliance & Legal"),

        # Infrastructure & Resources
        SeedSentence("I need a tool to predict maintenance needs for equipment before they break down.",
                     "Infrastructure & Resources"),
        SeedSentence("It would be great to track energy consumption and reduce costs automatically.",
                     "Infrastructure & Resources"),
        SeedSentence("I have problems scaling IT infrastructure without large upfront costs.",
                     "Infrastructure & Resources"),
        SeedSentence("I need software to manage and optimize office space for better productivity.",
                     "Infrastructure & Resources"),
        SeedSentence("It would be amazing to automate the process of allocating resources for new projects.",
                     "Infrastructure & Resources"),
        SeedSentence("I have problems managing storage space for physical and digital assets.",
                     "Infrastructure & Resources"),
        SeedSentence("I need a system to handle hardware upgrades without disrupting operations.",
                     "Infrastructure & Resources"),
    ]

    # Collect sentences
    sentence_data = semantic_search.collect_sentences()
    if not sentence_data:
        logger.error("No sentences collected. Exiting.")
        return

    # Perform search
    matched_sentences = semantic_search.search(sentence_data, seed_sentences)

    # Save results
    output_path = Path('matched_sentences.json')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(item) for item in matched_sentences], f,
                      ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    # Display sample results
    for item in matched_sentences[:10]:
        print("\nMatched Sentence:")
        print(f"Sentence: {item.sentence}")
        print(f"Category: {item.category}")
        print(f"Matched Seed: {item.matched_seed}")
        print(f"Similarity Score: {item.similarity_score:.4f}")
        print(f"Source: {item.source}")
        print(f"Author: {item.author}")
        print(f"Permalink: {item.permalink}")


if __name__ == '__main__':
    main()