from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.constants import CHAR_SUBSTITUTION_MAP, KEYBOARD_NEIGHBOR_MAP, load_common_emojis
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.types.char_level.case_change import CharCaseChange
from src.dataset_processing.perturbations.types.char_level.deletion import CharDeletion
from src.dataset_processing.perturbations.types.char_level.emoji import CharEmoji, EmojiConfig
from src.dataset_processing.perturbations.types.char_level.insertion import CharInsertion
from src.dataset_processing.perturbations.types.char_level.noise_insertion import CharNoiseInsertion, NoiseConfig
from src.dataset_processing.perturbations.types.char_level.repetition import CharRepetition
from src.dataset_processing.perturbations.types.char_level.replacement import CharReplacement, ReplacementConfig
from src.dataset_processing.perturbations.types.char_level.substitution import CharSubstitution, SubstitutionConfig
from src.dataset_processing.perturbations.types.char_level.swapping import CharSwapping
from src.dataset_processing.perturbations.types.word_level.deletion import WordDeletion
from src.dataset_processing.perturbations.types.word_level.insertion import WordInsertion
from src.dataset_processing.perturbations.types.word_level.internet_slang import InternetSlangInsertion
from src.dataset_processing.perturbations.types.word_level.repeat import WordRepetition
from src.dataset_processing.perturbations.types.word_level.swapping import WordSwapping
from src.dataset_processing.perturbations.types.word_level.synonym import SynonymReplacement
from src.dataset_processing.perturbations.types.word_level.taxonomy import NegativeTaxonomyPerturbation, PositiveTaxonomyPerturbation
from src.dataset_processing.perturbations.types.word_level.translation import PhraseTranslation


PERTURBATION_REGISTRY = {
    # Word-level perturbations
    PerturbationType.WORD_SYNONYM: SynonymReplacement,
    PerturbationType.WORD_INSERTION: WordInsertion,
    PerturbationType.WORD_DELETION: WordDeletion,
    PerturbationType.WORD_SWAPPING: WordSwapping,
    PerturbationType.WORD_REPETITION: WordRepetition,
    PerturbationType.WORD_INTERNET_SLANG: InternetSlangInsertion,
    PerturbationType.WORD_PHRASE_TRANSLATION: PhraseTranslation,
    
    # Taxonomy perturbations (with config)
    PerturbationType.WORD_TAXONOMY_POS: PositiveTaxonomyPerturbation,
    PerturbationType.WORD_TAXONOMY_NEG: NegativeTaxonomyPerturbation,
    
    # Character-level perturbations
    PerturbationType.CHAR_INSERTION: CharInsertion,
    PerturbationType.CHAR_DELETION: CharDeletion,
    PerturbationType.CHAR_REPLACEMENT: lambda config: CharReplacement(
        config,
        ReplacementConfig(keyboard_layout=KEYBOARD_NEIGHBOR_MAP)
    ),
    PerturbationType.CHAR_SWAPPING: CharSwapping,
    PerturbationType.CHAR_REPETITION: CharRepetition,
    PerturbationType.CHAR_SUBSTITUTION: lambda config: CharSubstitution(
        config,
        SubstitutionConfig(char_map=CHAR_SUBSTITUTION_MAP)
    ),
    PerturbationType.CHAR_INSERT_NOISE: lambda config: CharNoiseInsertion(
        config,
        NoiseConfig()
    ),
    PerturbationType.CHAR_CASE_CHANGE: CharCaseChange,
    PerturbationType.CHAR_EMOJI: lambda config: CharEmoji(
        config,
        EmojiConfig(emoji_loader=load_common_emojis)
    ),
}

def create_perturbation(
    perturbation_config: PerturbationConfig,
) -> TextPerturbation:
    """Factory function to create perturbation instances."""
    perturbation_class = PERTURBATION_REGISTRY.get(perturbation_config.type)
    if not perturbation_class:
        raise ValueError(f"Unknown perturbation type: {perturbation_config.type}")
    return perturbation_class(perturbation_config)