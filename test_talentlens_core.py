import unittest
from talentlens_core import find_skills, find_target_skills, score_resume, canonical_skill, predict_resume_category

class SkillMatchingTests(unittest.TestCase):
    def test_python_resume_and_target(self):
        resume = "Technical skills: Python, pandas and SQL."
        target = "Python"
        self.assertIn("python", find_skills(resume))
        self.assertEqual(find_target_skills(target), ["python"])
        self.assertEqual(score_resume(find_skills(resume), find_target_skills(target)), 100)

    def test_case_insensitive(self):
        self.assertEqual(find_target_skills("PYTHON"), ["python"])
        self.assertEqual(score_resume(find_skills("Experienced with PYTHON"), ["python"]), 100)

    def test_aliases(self):
        self.assertEqual(canonical_skill("sklearn"), "scikit-learn")
        self.assertIn("scikit-learn", find_skills("Used sklearn for model training"))
        self.assertIn("power bi", find_skills("Created dashboards in PowerBI"))

    def test_java_not_javascript(self):
        self.assertIn("javascript", find_skills("Built apps with JavaScript"))
        self.assertNotIn("java", find_skills("Built apps with JavaScript"))

    def test_no_known_target_skills_is_none_not_zero(self):
        self.assertIsNone(score_resume(["python"], find_target_skills("looking for a motivated person")))

    def test_missing_skill_score(self):
        self.assertEqual(score_resume(["python"], ["python", "sql"]), 50)



class CategoryModelTests(unittest.TestCase):
    def test_bundled_model_predicts_a_known_category(self):
        from pathlib import Path
        import joblib
        model = joblib.load(Path(__file__).with_name("resume_category_model.joblib"))
        text = ("Java Developer with 6 years of experience building enterprise applications using Java, J2EE, "
                "Spring Boot, Hibernate, REST APIs, microservices, Maven, SQL, Oracle, JDBC, JSP, Servlets, "
                "multithreading, design patterns, and object-oriented programming. Responsibilities include "
                "application architecture, code reviews, debugging, unit testing, deployment, performance "
                "tuning, and full software development lifecycle. Built scalable backend services and worked "
                "with cross-functional engineering teams.")
        prediction = predict_resume_category(text, model)
        self.assertEqual(prediction, "Java Developers/Architects Resumes")


if __name__ == "__main__":
    unittest.main()
