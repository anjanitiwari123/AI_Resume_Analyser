from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "all-MiniLM-L6-v2"
def load_matching_model():
    return SentenceTransformer(MODEL_NAME)
def calculate_semantic_score(resume_text, job_description, model):
    embeddings = model.encode(
        [resume_text, job_description],
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    score = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()
    return max(0, min(1, score)) * 100
def calculate_final_match_score(
        resume_text,
        job_description,
        matched_skills,
        missing_skills,
        model
):
    semantic_score = calculate_semantic_score(
        resume_text,
        job_description,
        model
    )
    total_skills = (
        len(matched_skills) +
        len(missing_skills)
    )
    if total_skills == 0:
        skill_score = 0
    else:
        skill_score = (
            len(matched_skills) / total_skills
        ) * 100
    final_score = (
        0.7 * skill_score +
        0.3 * semantic_score
    )
    return round(final_score, 1)