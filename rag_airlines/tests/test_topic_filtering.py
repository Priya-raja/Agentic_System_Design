import unittest
from context.indexers.retrieve import infer_question_topic, create_qdrant_filter


class TopicFilteringTests(unittest.TestCase):
    def test_known_topics(self):
        for question, topic in [('What meals are available?', 'menu'),
                                ('How much checked luggage?', 'baggage'),
                                ('What is the seat pitch?', 'seating')]:
            self.assertEqual(infer_question_topic(question), topic)
            filters = create_qdrant_filter(None, topic)
            self.assertEqual(filters.must[0].key, 'metadata.topic')
            self.assertEqual(filters.must[0].match.value, topic)

    def test_unknown_and_multiple_topics_are_unfiltered(self):
        for question in ['Can I get accommodation if delayed?',
                         'What are baggage and meal rules?',
                         'What is the advantage?']:
            self.assertIsNone(infer_question_topic(question))
        self.assertIsNone(create_qdrant_filter(None, None))

    def test_year_filter_preserved(self):
        filters = create_qdrant_filter(2025, 'baggage')
        self.assertEqual({condition.key: condition.match.value for condition in filters.must},
                         {'metadata.policy_year': 2025, 'metadata.topic': 'baggage'})
