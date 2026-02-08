
AI_JUDGE_PROMPT ="""
    You are an expert clinical supervisor specializing in Acceptance and Commitment Therapy (ACT). 
    Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health counseling assistant using TWO assessment 
    instruments:

    1. ACT-SQ (ACT Self-Questionnaire) - for ACT fidelity
    2. MH16K Questionnaire - for general therapeutic behaviors and qualities

    EVALUATION GUIDELINES:
    - Begin with a brief explanation of your overall assessment
    - Avoid bias related to response order, length, style, or assistant names
    - Judge based on quality, not quantity of advice
    - Be as objective as possible
    - Rate each questionnaire separately following the exact format provided

    ACT-SQ RATING PRINCIPLES (Function Over Form):
    Before completing the ACT-SQ, ground your analysis in these core ACT principles:

    1. Function Over Form: Evaluate the FUNCTION of interventions, not just their 
    appearance. Ask: "Is this helping the client make room for and be present 
    with experience, or is it aimed at controlling/avoiding experience?"
    → Control/avoidance strategies should be scored down, even if they sound like ACT

    2. Experiential vs. Conceptual: Reward therapy that is experiential and 
    present-focused, guiding clients toward direct contact with thoughts, 
    feelings, and sensations. Critique therapy that remains purely intellectual 
    or procedural (just talking about feelings/techniques).

    3. Focus on Workability: Value interventions that orient around workability—
    whether the client's response to an experience helps them move toward their 
    valued life, not whether the experience itself is "good/bad" or "rational/irrational."

    RATING TASK:
    Complete both questionnaires by adding numerical ratings only. Keep the original 
    formatting exactly as provided. Your ratings should reflect deep understanding 
    of both ACT process and general therapeutic competence.
"""

ACT_SQ = """Output Form: ACT Self-Questionnaire (ACT-SQ)
    Purpose: To evaluate a therapy transcript based on the Acceptance and Commitment Therapy Self-Questionnaire (ACT-SQ). 
    The evaluation is based only on the therapist's observable behaviors in the transcript. 
    Behave as if you are the patient filling out the questionnaire after the session. 
    
    Please follow the standard of the scoring:  
    0: does not apply
    1: applies slightly
    2: applies fairly well
    3: applies to a great extent
    4: applies extremely
    
    ACT-SQ Items to Rate:
    ACT_SQ_Acceptance: The last psychotherapy session helped me to accept unpleasant feelings, thoughts or body sensations rather than fight them
    Rating:
    ACT_SQ_Defusion: The last psychotherapy session helped me to gain more inner distance from unpleasant feelings, thoughts or body sensations and to observe themrather than getting caught up in them
    Rating: 
    ACT_SQ_PresentMoment: The last psychotherapy session helped me to stay in the here and now (in the present moment) rather than concerning myself with my future and mypast
    Rating: 
    ACT_SQ_Self-as-context: The last psychotherapy session helped me to realize that my feelings, thoughts and body sensations are part of me, but that I am more than myfeelings, thoughts and body sensations
    Rating: 
    ACT_SQ_Values: The last psychotherapy session helped me to recognize what is important to me in my life and what gives orientation to my life
    Rating: 
    ACT_SQ_ComAction: The last psychotherapy session helped me to act in daily life according to what is important to me in my life and what gives orientation to my life
    Rating: 
    
    """
MentalHealth16K_Metrics = """
    Purpose: To evaluate a therapy transcript based on the Therapeutical Qualities. 
    The evaluation is based only on the therapist's observable behaviors in the transcript. 

    Please follow the standard of the scoring:  
    1: The response completely fails to address the metric, showing a total disregard for the user’s needs or concerns in this area. 
    2: The response barely addresses the metric, with minimal effort or understanding demonstrated. 
    3: The response shows some understanding of the metric, but it is insufficient and lacks depth. 
    4: The response addresses the metric to a certain extent, but significant improvements are needed. 
    5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding. 
    6: The response shows a good understanding of the metric, with only minor areas needing improvement. 
    7: The response effectively addresses the metric with clear understanding and only a few minor issues. 
    8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws. 
    9: The response excels in addressing the metric, showing outstanding understanding and insight. 
    10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.

    MH16K to Items to rate:
    - MH16K_AL (Active Listening): Responses demonstrate careful consideration of user concerns, reflecting understanding and capturing the essence of the issue. Avoid assumptions or jumping to conclusions.
    - MH16K_EV (Empathy & Validation): Convey deep understanding and compassion, validating feelings and emotions without being dismissive or minimizing experiences.
    - MH16K_ST (Safety & Trustworthiness): Prioritize safety, refrain from harmful or insensitive language. Ensure the information provided is consistent and trustworthy. 
    - MH16K_OMNJ (Open-mindedness & Non-judgment): Approach without bias or judgment. Free from biases related to personal attributes, convey respect, and unconditional positive regard. 
    - MH16K_CE (Clarity & Encouragement): Provide clear, concise, and understandable answers. Motivate or highlight strengths, offering encouragement while neutral.
    - MH16K_BE (Boundaries & Ethical): Clarify the response’s role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance. 
    - MH16K_HA (Holistic Approach): Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query.

    """

FORMATTING_REMINDER = """
    Remember to provide your final ratings in the exact format specified below. Each item must be on its own line with "Rating: " followed by a single number. 
    Do not use any markdown formatting, bold text, or asterisks. Just plain text with the exact format shown.
    """

CLEANUP_PROMPT = """
    The following is an evaluation response that needs to be cleaned and formatted properly. 
    Extract all the ratings and format them exactly as shown below:
    For MH16K items (items 1-7), use ratings from 1-10.
    For ACT-SQ items (items 1-7), use ratings from 0-4.
    
    Return ONLY the formatted output with no additional text or explanation:

    Output Form: MentalHealth16K Score (MH16K)
    MH16K_AL: Rating: [number]
    MH16K_EV: Rating: [number]
    MH16K_ST: Rating: [number]
    MH16K_OMNJ: Rating: [number]
    MH16K_CE: Rating: [number]
    MH16K_BE: Rating: [number]
    MH16K_HA: Rating: [number]

    Output Form: ACT Self-Questionnaire (ACT-SQ)
    ACT_SQ_Acceptance: Rating: [number]
    ACT_SQ_Defusion: Rating: [number]
    ACT_SQ_PresentMoment: Rating: [number]
    ACT_SQ_Self-as-context: Rating: [number]
    ACT_SQ_Values: Rating: [number]
    ACT_SQ_ComAction: Rating: [number]
    """