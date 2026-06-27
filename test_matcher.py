"""
Unit and integration tests for the resume-jd-matcher.

Run:  python -m unittest -v
"""

import math
import unittest

import text_utils
import tfidf
from analyzer import analyze, detect_skills, grade
from skills_db import build_alias_index


class TextUtilsTests(unittest.TestCase):
    def test_tokenize_keeps_symbolic_tokens(self):
        toks = text_utils.tokenize("I use C++, C#, Node.js and CI/CD daily.")
        self.assertIn("c++", toks)
        self.assertIn("c#", toks)
        self.assertIn("node.js", toks)
        self.assertIn("ci/cd", toks)

    def test_content_tokens_remove_stopwords(self):
        toks = text_utils.content_tokens("the model is very good and fast")
        self.assertNotIn("the", toks)
        self.assertNotIn("is", toks)
        self.assertIn("model", toks)
        self.assertIn("fast", toks)

    def test_phrase_set_contains_trigram(self):
        phrases = text_utils.phrase_set("natural language processing pipelines")
        self.assertIn("natural language processing", phrases)
        self.assertIn("language processing pipelines", phrases)


class TfidfTests(unittest.TestCase):
    def test_identical_text_similarity_is_one(self):
        text = "python machine learning pytorch nlp docker"
        self.assertAlmostEqual(tfidf.text_similarity(text, text), 1.0, places=6)

    def test_disjoint_text_similarity_is_zero(self):
        a = "python machine learning"
        b = "gardening cooking baking"
        self.assertEqual(tfidf.text_similarity(a, b), 0.0)

    def test_similarity_is_symmetric_and_bounded(self):
        a = "python sql docker machine learning"
        b = "machine learning python deployment"
        s1 = tfidf.text_similarity(a, b)
        s2 = tfidf.text_similarity(b, a)
        self.assertAlmostEqual(s1, s2, places=9)
        self.assertTrue(0.0 <= s1 <= 1.0)

    def test_cosine_handles_empty_vectors(self):
        self.assertEqual(tfidf.cosine_similarity({}, {"a": 1.0}), 0.0)

    def test_idf_rewards_rare_terms(self):
        docs = [["python", "common"], ["java", "common"], ["go", "common"]]
        idf = tfidf.inverse_document_frequencies(docs)
        self.assertGreater(idf["python"], idf["common"])


class SkillDetectionTests(unittest.TestCase):
    def test_alias_index_resolves_abbreviations(self):
        index = build_alias_index()
        self.assertEqual(index["ml"], "Machine Learning")
        self.assertEqual(index["js"], "JavaScript")
        self.assertEqual(index["k8s"], "Kubernetes")

    def test_detect_skills_finds_multiword_phrase(self):
        found = detect_skills("We run natural language processing on big data.")
        self.assertIn("NLP", found)

    def test_detect_skills_alias_equivalence(self):
        # "JS" in one doc and "JavaScript" in another resolve to the same skill.
        self.assertEqual(detect_skills("strong JS skills"),
                         detect_skills("strong JavaScript skills"))

    def test_detect_skills_ignores_unknown_words(self):
        found = detect_skills("the quick brown fox jumps")
        self.assertEqual(found, set())


class AnalyzeIntegrationTests(unittest.TestCase):
    def setUp(self):
        with open("samples/resume_sample.txt", encoding="utf-8") as fh:
            self.resume = fh.read()
        with open("samples/jd_sample.txt", encoding="utf-8") as fh:
            self.jd = fh.read()

    def test_report_structure_and_bounds(self):
        rep = analyze(self.resume, self.jd)
        self.assertTrue(0.0 <= rep.score <= 100.0)
        self.assertTrue(0.0 <= rep.skill_coverage <= 100.0)
        self.assertTrue(0.0 <= rep.text_similarity <= 100.0)

    def test_matched_and_missing_are_disjoint(self):
        rep = analyze(self.resume, self.jd)
        self.assertEqual(set(rep.matched_skills) & set(rep.missing_skills), set())

    def test_known_overlap_detected(self):
        rep = analyze(self.resume, self.jd)
        # Resume and JD both mention these → must be matched.
        for skill in ("Python", "Machine Learning", "SQL"):
            self.assertIn(skill, rep.matched_skills)

    def test_known_gaps_detected(self):
        rep = analyze(self.resume, self.jd)
        # JD asks for these; sample resume omits them → must be missing.
        for skill in ("PyTorch", "Docker", "NLP"):
            self.assertIn(skill, rep.missing_skills)

    def test_suggestions_target_missing_skills(self):
        rep = analyze(self.resume, self.jd)
        self.assertTrue(rep.suggestions)
        # Every suggestion is tagged with a skill that is actually missing.
        for s in rep.suggestions:
            tag = s[1:s.index("]")]
            self.assertIn(tag, rep.missing_skills)

    def test_identical_documents_score_high(self):
        rep = analyze(self.jd, self.jd)
        self.assertGreaterEqual(rep.score, 95.0)
        self.assertEqual(rep.missing_skills, [])

    def test_grade_bands(self):
        self.assertEqual(grade(85), "Strong match")
        self.assertEqual(grade(10), "Weak match — significant tailoring needed")


if __name__ == "__main__":
    unittest.main()
