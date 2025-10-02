from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text
text = """
Leader of opposition in the Lok Sabha, Rahul Gandhi, accused the BJP-led central government of mounting a “wholesale attack on Indian democracy” during an interaction at EIA University in Colombia. The BJP hit back, calling him a “Leader of Propaganda” and criticising him for "once again" denigrating Indian democracy on a foreign platform.
"The single biggest risk is the attack on democracy that is taking place in India because India has multiple religions, multiple traditions, multiple languages. And India actually is a conversation between all its people and different traditions, different religions, different ideas require space. And that the best method for creating that space is the democratic system. And currently there is a wholesale attack on the democratic system in India," Rahul said during a recent interaction with students.
"""

# Summarize
summary = summarizer(text, max_length=150, min_length=25, do_sample=False)

print("Summary:", summary[0]['summary_text'])
