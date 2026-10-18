import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pathlib import Path
import argparse
import json

BASE_DIR = Path(__file__).resolve().parent
try:
    nlp = spacy.load("en_core_web_sm")
except OSError as exc:
    raise SystemExit("Missing spaCy model. Run: python -m spacy download en_core_web_sm") from exc
analyzer = SentimentIntensityAnalyzer()

# LOAD NRC AFFECT INTENSITY LEXICON
affect_lexicon = {}

LEXICON_PATH = BASE_DIR / "NRC-Emotion-Intensity-Lexicon-v1.txt"
if not LEXICON_PATH.exists():
    raise SystemExit(
        f"Missing NRC lexicon at {LEXICON_PATH}. Place your licensed/downloaded lexicon beside main.py."
    )

with LEXICON_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        fields = line.strip().split("\t")
        if len(fields) != 3:
            continue  # skip blank or non-data lines
        word, emotion, intensity = fields
        try:
            intensity = float(intensity)
        except ValueError:
            continue  # skip header
        affect_lexicon.setdefault(word.lower(), []).append(intensity)

# INTENSIFIERS
intensifiers = {
    "absolutely", "altogether", "completely", "critically",
    "deeply", "desperately", "dramatically", "drastically",
    "enormously", "entirely", "especially", "exceedingly",
    "exceptionally", "extremely", "fully", "highly",
    "horribly", "immensely", "incredibly", "intensely",
    "particularly", "perfectly", "profoundly", "remarkably",
    "severely", "strongly", "substantially", "terribly",
    "thoroughly", "totally", "tremendously", "unmistakably",
    "unquestionably", "unusually", "utterly", "very",
    "wholly", "super"
}

# URGENCY / ALARM
immediacy_words = {
    "now", "today", "tonight", "immediately",
    "imminent", "soon", "currently"
}

immediacy_phrases = {
    "right now",
    "within hours",
    "within days",
    "at once"
}

threat_phrases = {
    "poses a threat",
    "poses a serious threat",
    "poses a major threat",
    "faces a threat",
    "under threat",
    "at immediate risk",
    "at serious risk",
    "at severe risk",
    "in imminent danger",
    "faces imminent danger",
    "be catastrophic",
    "prove fatal",
    "have devastating consequences",
    "cause widespread damage",
    "trigger a crisis",
    "result in serious harm",
    "is putting lives at risk",
    "puts lives at risk"
}

# SENSATIONAL VOCABULARY


def analyze_article(text, verbose=False):

    doc = nlp(text)

    # WORD COUNT
    word_count = sum(1 for token in doc if token.is_alpha)

    # EMOTIONAL INTENSITY
    emotional_words = []
    emotion_intensity_total = 0

    for token in doc:
        word = token.lemma_.lower()

        if word in affect_lexicon:
            intensities = affect_lexicon[word]
            intensity = sum(intensities) / len(intensities)

            emotional_words.append((word, intensity))
            emotion_intensity_total += intensity

    if word_count > 0:
        emotional_intensity = emotion_intensity_total / word_count
    else:
        emotional_intensity = 0
    # SENTIMENT
    sentiment = analyzer.polarity_scores(text)

    # SUBJECTIVITY
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    # EVALUATIVE MODIFIERS
    evaluative_modifiers = []

    for token in doc:
        if token.pos_ in {"ADJ", "ADV"} and token.dep_ in {
            "amod", "acomp", "advmod"
        }:
            word = token.lemma_.lower()

            if word in affect_lexicon:
                evaluative_modifiers.append(token)

    evaluative_modifier_density = (
        len(evaluative_modifiers) / word_count
        if word_count else 0
    )

    # EXAGGERATION
    exaggeration_markers = []

    absolute_words = {
        # Universal inclusion
        "all", "every", "each", "both",
        "everyone", "everybody", "everything", "everywhere",

        # Total exclusion
        "no", "none", "neither", "nobody", "nothing", "nowhere",

        # Absolute frequency
        "always", "never", "constantly", "continually", "forever"
    }

    for token in doc:
        word = token.lemma_.lower()

        # Superlatives
        if token.tag_ in {"JJS", "RBS"}:
            exaggeration_markers.append(token)

        # Absolute / totalizing language
        elif word in absolute_words:
            exaggeration_markers.append(token)

    exaggeration_density = (
        len(exaggeration_markers) / word_count
        if word_count else 0
    )

    # URGENCY / ALARM
    immediacy_markers = []
    threat_markers = []

    for token in doc:
        word = token.lemma_.lower()

        if word in immediacy_words:
            immediacy_markers.append(token)

    text_lower = text.lower()

    for phrase in immediacy_phrases:
        if phrase in text_lower:
            immediacy_markers.append(phrase)

    for phrase in threat_phrases:
        if phrase in text_lower:
            threat_markers.append(phrase)

    immediacy_density = len(immediacy_markers) / word_count if word_count else 0
    threat_density = len(threat_markers) / word_count if word_count else 0
    
    # CAUSAL FRAMING
    # Strong causal certainty + consequential outcome + heightened connotation

    direct_causal_verbs = {
        "cause",
        "caused",
        "causes",
        "trigger",
        "triggered",
        "triggers",
        "lead",
        "led",
        "leads",
        "result",
        "resulted",
        "results"
    }

    hedging_words = {
        "may",
        "might",
        "could",
        "can",
        "possibly",
        "potentially",
        "likely",
        "unlikely",
        "appears",
        "appeared",
        "suggests",
        "suggested",
        "seems",
        "seemed",
        "would",
        "should"
    }

    def has_causal_hedge(sent, token):
        """
        Detect explicit uncertainty or conditionality around the
        causal relationship without being overly conservative.
        """

        for other in sent:
            if abs(other.i - token.i) <= 4:
                if other.lemma_.lower() in hedging_words:
                    return True

        for other in sent:
            if other.i < token.i and other.lower_ == "if":
                return True

        return False

    def get_causal_outcome_tokens(token):
        """
        Identify the consequence of a causal relationship
        using only direct grammatical complements.
        """

        outcome_roots = []

        if token.lemma_.lower() in {"cause", "trigger"}:
            for child in token.children:
                if child.dep_ in {"obj", "dobj", "xcomp"}:
                    outcome_roots.append(child)

        elif token.lemma_.lower() == "lead":
            for child in token.children:
                if child.dep_ == "prep" and child.lemma_.lower() == "to":
                    for prep_child in child.children:
                        if prep_child.dep_ in {"pobj", "obj"}:
                            outcome_roots.append(prep_child)

        elif token.lemma_.lower() == "result":
            for child in token.children:
                if child.dep_ == "prep" and child.lemma_.lower() == "in":
                    for prep_child in child.children:
                        if prep_child.dep_ in {"pobj", "obj"}:
                            outcome_roots.append(prep_child)

        outcome_tokens = set(outcome_roots)

        for root in outcome_roots:
            for child in root.children:
                if child.dep_ in {
                    "amod",
                    "compound",
                    "nummod",
                    "det",
                    "poss"
                }:
                    outcome_tokens.add(child)

        return outcome_tokens
    def get_affect_intensity(token):
        """
        Return the strongest emotion intensity associated with a token.
        """

        if not token.is_alpha:
            return 0

        word = token.lemma_.lower()
        intensities = affect_lexicon.get(word, [])

        if not intensities:
            return 0

        return max(intensities)
    def get_responsibility_tokens(sent):
        """
        Identify explicit responsibility attribution using
        grammatical relationships rather than a fixed vocabulary.

        The attribution itself is not treated as sensational.
        It becomes relevant only when the attributed outcome
        is linguistically heightened.
        """

        responsibility_tokens = set()

        responsibility_patterns = {
            "blame",
            "accuse",
            "fault",
            "credit",
            "praise",
            "responsible",
        }

        for token in sent:
            word = token.lemma_.lower()

            # Explicit responsibility / credit constructions
            if word in responsibility_patterns:

                responsibility_tokens.add(token)

                # Actor or target connected to the attribution
                for child in token.children:
                    if child.dep_ in {
                        "nsubj",
                        "nsubjpass",
                        "dobj",
                        "obj",
                        "agent",
                        "prep",
                        "pobj"
                    }:
                        responsibility_tokens.add(child)

                # Nearby evaluative or emotionally heightened language
                for other in sent:
                    if abs(other.i - token.i) <= 4:
                        intensity = get_affect_intensity(other)

                        if intensity > 0:
                            responsibility_tokens.add(other)

            # Adjectival responsibility constructions:
            # "X is responsible for Y"
            if word == "responsible":

                for child in token.children:
                    if child.dep_ in {"prep", "attr"}:
                        responsibility_tokens.add(child)

                        for subchild in child.children:
                            if subchild.dep_ in {
                                "pobj",
                                "obj",
                                "dobj"
                            }:
                                responsibility_tokens.add(subchild)

        return responsibility_tokens
    def get_prediction_outcome_tokens(token):
        """
        Identify the proposition governed by a modal auxiliary.

        The system uses grammatical structure rather than a list
        of prediction words. The modal must govern a main verb,
        which represents the projected or conditional event.
        """

        outcome_tokens = set()

        for child in token.children:
            if child.pos_ in {"VERB", "AUX"} and child.dep_ in {
                "xcomp",
                "ccomp"
            }:
                outcome_tokens.add(child)

                for descendant in child.subtree:
                    if descendant.pos_ in {
                        "NOUN",
                        "PROPN",
                        "ADJ",
                        "ADV",
                        "VERB",
                        "AUX"
                    }:
                        outcome_tokens.add(descendant)

        return outcome_tokens
    def get_prediction_tokens(sent):
        """
        Identify predictive or speculative constructions using
        syntactic structure rather than a fixed trigger-word list.

        A modal auxiliary is treated only as a candidate when it
        governs a proposition. The proposition is then evaluated
        for linguistic heightening.
        """

        prediction_tokens = set()

        for token in sent:

            if token.tag_ != "MD":
                continue

            outcome_tokens = get_prediction_outcome_tokens(token)

            if not outcome_tokens:
                continue

            prediction_tokens.add(token)
            prediction_tokens.update(outcome_tokens)

        return prediction_tokens
    prediction_scores = []

    for sent in doc.sents:
        prediction_tokens = get_prediction_tokens(sent)

        if not prediction_tokens:
            continue

        heightened_values = []

        for token in prediction_tokens:
            intensity = get_affect_intensity(token)

            if intensity > 0:
                heightened_values.append(intensity)

        if not heightened_values:
            continue

        prediction_peak = max(heightened_values)

        prediction_mean = (
            sum(heightened_values) /
            len(heightened_values)
        )

        prediction_salience = (
            prediction_peak * 0.6 +
            prediction_mean * 0.4
        )

        context_affect_values = []

        for token in sent:
            if token in prediction_tokens:
                continue

            intensity = get_affect_intensity(token)

            if intensity > 0:
                context_affect_values.append(intensity)

        if context_affect_values:
            context_baseline = (
                sum(context_affect_values) /
                len(context_affect_values)
            )
        else:
            context_baseline = 0

        heightened_difference = max(
            0,
            prediction_salience - context_baseline
        )

        framing_strength = min(
            heightened_difference / 0.25,
            1.0
        )

        prediction_scores.append(
            prediction_salience *
            framing_strength
        )
    if prediction_scores:
        prediction_sensationalism_score = (
            sum(prediction_scores) /
            len(prediction_scores)
        )
    else:
        prediction_sensationalism_score = 0


    responsibility_scores = []

    for sent in doc.sents:
        responsibility_tokens = get_responsibility_tokens(sent)

        if not responsibility_tokens:
            continue

        heightened_values = []

        for token in responsibility_tokens:
            intensity = get_affect_intensity(token)

            if intensity > 0:
                heightened_values.append(intensity)

        if not heightened_values:
            continue

        responsibility_peak = max(heightened_values)

        responsibility_mean = (
            sum(heightened_values) /
            len(heightened_values)
        )

        responsibility_salience = (
            responsibility_peak * 0.6 +
            responsibility_mean * 0.4
        )

        context_affect_values = []

        for token in sent:
            if token in responsibility_tokens:
                continue

            intensity = get_affect_intensity(token)

            if intensity > 0:
                context_affect_values.append(intensity)

        if context_affect_values:
            context_baseline = (
                sum(context_affect_values) /
                len(context_affect_values)
            )
        else:
            context_baseline = 0

        heightened_difference = max(
            0,
            responsibility_salience - context_baseline
        )

        framing_strength = min(
            heightened_difference / 0.25,
            1.0
        )

        responsibility_scores.append(
            responsibility_salience *
            framing_strength
        )

    if responsibility_scores:
        responsibility_sensationalism_score = (
            sum(responsibility_scores) /
            len(responsibility_scores)
        )
    else:
        responsibility_sensationalism_score = 0

 

    strong_causal_count = 0
    causal_candidate_count = 0
    causal_relationship_scores = []

    for sent in doc.sents:
        for token in sent:

            if token.pos_ != "VERB":
                continue

            if token.lemma_.lower() not in direct_causal_verbs:
                continue

            causal_candidate_count += 1
            outcome_tokens = get_causal_outcome_tokens(token)

            if not outcome_tokens:
                continue

            if has_causal_hedge(sent, token):
                continue

            strong_causal_count += 1

            outcome_affect_values = []

            for outcome_token in outcome_tokens:
                intensity = get_affect_intensity(outcome_token)

                if intensity > 0:
                    outcome_affect_values.append(intensity)

            if not outcome_affect_values:
                continue

            outcome_peak = max(outcome_affect_values)
            outcome_mean = sum(outcome_affect_values) / len(outcome_affect_values)

            consequence_salience = (
                outcome_peak * 0.6 +
                outcome_mean * 0.4
            )

            context_affect_values = []

            for context_token in sent:
                if context_token in outcome_tokens:
                    continue

                if context_token == token:
                    continue

                intensity = get_affect_intensity(context_token)

                if intensity > 0:
                    context_affect_values.append(intensity)

            if context_affect_values:
                context_baseline = (
                    sum(context_affect_values) /
                    len(context_affect_values)
                )
            else:
                context_baseline = 0

            heightened_difference = max(
                0,
                consequence_salience - context_baseline
            )

            framing_strength = min(
                heightened_difference / 0.25,
                1.0
            )

            relationship_score = (
                consequence_salience *
                framing_strength
            )

            causal_relationship_scores.append(
                relationship_score
            )

    if causal_relationship_scores:
        causal_sensationalism_score = (
            sum(causal_relationship_scores) /
            len(causal_relationship_scores)
        )
    else:
        causal_sensationalism_score = 0



    # INTENSIFIERS
    intensifier_words = []

    for token in doc:
        word = token.lemma_.lower()

        if word in intensifiers:
            intensifier_words.append(word)

    intensifier_density = (
        len(intensifier_words) / word_count
        if word_count else 0
    )

    # PUNCTUATION
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # SALIENCE — INFORMATION FRAMING
    #
    # Goal:
    # Measure whether an article repeatedly foregrounds a small number
    # of entities, especially when those entities also appear in the headline.
    #
    # This is NOT simply "number of named entities."
    # We are looking for concentration + repetition + headline prominence.

    # ---------------------------------------------------------
    # ENTITY CLEANING
    # ---------------------------------------------------------

    allowed_entity_labels = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "EVENT",
        "FAC"
    }

    metadata_phrases = {
        "purchase licensing rights",
        "terms of use",
        "privacy policy",
        "our standards",
        "item",
        "updated",
        "gmt",
        "copyright",
        "subscribe",
        "sign up",
        "newsletter",
        "read more",
        "follow us",
        "cookie",
        "cookies"
    }

    def clean_entity(entity):
        entity = entity.lower().strip()

        # Remove leading/trailing punctuation
        entity = entity.strip(".,:;!?()[]{}\"'")

        # Collapse whitespace
        entity = " ".join(entity.split())

        # Remove leading "the" so:
        # "the supreme court" -> "supreme court"
        if entity.startswith("the "):
            entity = entity[4:]
        # Normalize possessive forms:
        # "Hansen's" -> "Hansen"
        # "Walker’s" -> "Walker"
        if entity.endswith("'s") or entity.endswith("’s"):
            entity = entity[:-2]

        return entity

    def is_valid_entity(entity):
        if len(entity) < 3:
            return False

        # Avoid entities containing obvious webpage noise
        if any(phrase in entity for phrase in metadata_phrases):
            return False

        # Avoid timestamps / dates / numeric-heavy entities
        digit_count = sum(char.isdigit() for char in entity)

        if digit_count > 0:
            return False

        # Avoid entities that are basically symbols
        alphabetic_count = sum(char.isalpha() for char in entity)

        if alphabetic_count < 3:
            return False

        return True

    # ---------------------------------------------------------
    # COLLECT BODY ENTITIES
    # ---------------------------------------------------------

    body_entities = []

    for ent in doc.ents:

        if ent.label_ not in allowed_entity_labels:
            continue

        entity = clean_entity(ent.text)

        if not is_valid_entity(entity):
            continue

        body_entities.append(entity)

    # ---------------------------------------------------------
    # ENTITY FREQUENCY
    # ---------------------------------------------------------

    entity_counts = {}

    for entity in body_entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1

    # ---------------------------------------------------------
    # REMOVE ONE-OFF ENTITIES
    #
    # A single mention is usually not meaningful evidence of
    # sustained information salience.
    # ---------------------------------------------------------

    repeated_entities = {
        entity: count
        for entity, count in entity_counts.items()
        if count >= 2
    }

    # ---------------------------------------------------------
    # NORMALIZED ENTITY FREQUENCY
    # ---------------------------------------------------------

    total_entity_mentions = sum(entity_counts.values())

    if total_entity_mentions > 0:
        entity_share = {
            entity: count / total_entity_mentions
            for entity, count in entity_counts.items()
        }
    else:
        entity_share = {}
    # ---------------------------------------------------------
    # ENTITY CONCENTRATION
    #
    # How much of the repeated entity attention is concentrated
    # in the top entities?
    #
    # Example:
    # 1 entity = 70% of mentions
    # -> strong concentration
    #
    # 15 entities each = ~6%
    # -> weak concentration
    # ---------------------------------------------------------

    repeated_entity_share = {
        entity: count / total_entity_mentions
        for entity, count in repeated_entities.items()
    }

    sorted_entity_shares = sorted(
        repeated_entity_share.values(),
        reverse=True
    )

    top_3_share = sum(sorted_entity_shares[:3])

    # ---------------------------------------------------------
    # HEADLINE EXTRACTION
    # ---------------------------------------------------------
    #
    # We use the first line as the headline because the current
    # article text files contain the headline before the body.
    #
    # If the files later use another format, this can be changed.
    # ---------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    headline = lines[0] if lines else ""

    headline_doc = nlp(headline)

    headline_entities = set()

    for ent in headline_doc.ents:

        if ent.label_ not in allowed_entity_labels:
            continue

        entity = clean_entity(ent.text)

        if not is_valid_entity(entity):
            continue

        headline_entities.add(entity)

    # ---------------------------------------------------------
    # HEADLINE-BODY OVERLAP
    # ---------------------------------------------------------

    headline_body_entities = (
        headline_entities.intersection(repeated_entities.keys())
    )

    # ---------------------------------------------------------
    # HEADLINE PROMINENCE
    #
    # Of the entities that the headline mentions, how many are
    # actually among the article's repeatedly foregrounded entities?
    # ---------------------------------------------------------

    if headline_entities:
        headline_prominence = (
            len(headline_body_entities)
            / len(headline_entities)
        )
    else:
        headline_prominence = 0

    # HEADLINE ENTITY BODY SHARE

    if headline_body_entities:
        headline_body_share = max(
            entity_share.get(entity, 0)
            for entity in headline_body_entities
        )
    else:
        headline_body_share = 0

    # ---------------------------------------------------------
    # REPEATED FOREGROUNDING
    #
    # How much repeated entity attention is concentrated in
    # the top 3 entities?
    # ---------------------------------------------------------

    repeated_foregrounding = min(top_3_share, 1.0)

    # ---------------------------------------------------------
    # SALIENCE SCORE
    #
    # This is deliberately conservative.
    #
    # Concentration = main signal
    # Headline prominence = supporting signal
    # Headline/body share = supporting signal
    #
    # We do NOT reward an article simply for having many entities.
    # ---------------------------------------------------------

    salience_score = (
        repeated_foregrounding * 50 +
        headline_prominence * 25 +
        headline_body_share * 25
    )

    salience_score = min(salience_score, 100)
    # NORMALIZED FEATURE SCORES
    sentiment_score = abs(sentiment["compound"]) * (1 - sentiment["neu"]) * 100
    subjectivity_score = subjectivity * 100
    emotion_score = min(emotional_intensity * 100, 100)
    evaluative_modifier_score = min(evaluative_modifier_density * 100, 100)
    intensifier_score = min(intensifier_density * 100, 100)
    immediacy_score = min(immediacy_density * 100, 100)
    threat_score = min(threat_density * 100, 100)

    punctuation_score = min(
        ((exclamation_count + question_count) / word_count) * 100
        if word_count else 0,
        100
    )

        # SKEETERSCORE

    # The original six weights add to 0.85, not 1.00.
    # Renormalize the SAME six features without silently introducing new weights.
    raw_weighted_score = (
        sentiment_score * 0.20 +
        subjectivity_score * 0.20 +
        emotion_score * 0.20 +
        evaluative_modifier_score * 0.10 +
        intensifier_score * 0.10 +
        punctuation_score * 0.05
    )
    skeeter_score = raw_weighted_score / 0.85

    result = {
        "skeeter_score": round(skeeter_score, 3),
        "word_count": word_count,
        "scored_features": {
            "sentiment": round(sentiment_score, 3),
            "subjectivity": round(subjectivity_score, 3),
            "emotion": round(emotion_score, 3),
            "evaluative_modifiers": round(evaluative_modifier_score, 3),
            "intensifiers": round(intensifier_score, 3),
            "punctuation": round(punctuation_score, 3),
        },
        "experimental_features_not_in_total": {
            "immediacy": round(immediacy_score, 3),
            "threat": round(threat_score, 3),
            "exaggeration_density": round(exaggeration_density, 5),
            "causal_framing": round(causal_sensationalism_score, 3),
            "responsibility_framing": round(responsibility_sensationalism_score, 3),
            "prediction_framing": round(prediction_sensationalism_score, 3),
            "entity_salience": round(salience_score, 3),
        },
    }
    if verbose:
        print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze sensational language in text articles.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Article .txt files; default: all .txt files in articles/")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    paths = args.paths or sorted((BASE_DIR / "articles").glob("*.txt"))
    if not paths:
        parser.error("No articles found. Put .txt files in articles/ or pass a file path.")
    outputs = []
    for path in paths:
        if not path.is_file():
            parser.error(f"Article file not found: {path}")
        result = analyze_article(path.read_text(encoding="utf-8"))
        outputs.append({"file": str(path), **result})
    if args.json:
        print(json.dumps(outputs, indent=2))
    else:
        for item in outputs:
            print(f"{Path(item['file']).name}: {item['skeeter_score']:.2f}/100 "
                  f"({item['word_count']} words)")
            for feature, score in item['scored_features'].items():
                print(f"  {feature}: {score:.2f}")
            print("  Experimental features: available with --json (not part of score)")


if __name__ == "__main__":
    main()
