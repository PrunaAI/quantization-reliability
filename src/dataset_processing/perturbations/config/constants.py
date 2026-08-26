def load_common_emojis():
    """Load the 100 most commonly used emojis."""
    COMMON_EMOJIS = [
        # Smileys & Emotion
        "😂", "❤️", "🤣", "👍", "😭", 
        "🙏", "😘", "🥰", "😍", "😊",
        "😅", "🔥", "😁", "😏", "❤️‍🔥",
        "💋", "🤦", "🥺", "😳", "💕",
        "😒", "🙄", "💜", "😩", "✨",
        "😔", "👀", "🎉", "💪", "☺️",
        "💖", "😌", "🙂", "🤗", "💙",
        "🤔", "😄", "😪", "🥱", "😴",
        "🤤", "😑", "🌹", "🥲", "🫶",
        "😡", "🤢", "😎", "😋", "😉",
        
        # Gestures & People
        "🤷", "🙌", "👋", "🤝", "👏",
        "🎊", "🫂", "👌", "🤌", "✌️",
        "🤘", "🫡", "🦾", "🖐️", "🤚",
        "🫴", "👊", "🤜", "🤛", "🙋",
        
        # Hearts & Love
        "💗", "💓", "💞", "💝", "❣️",
        "💌", "💘", "💟", "💔", "💑",
        
        # Nature & Objects
        "🌸", "🌺", "🌷", "🌟", "⭐",
        "🌙", "⚡", "💫", "🌈", "🍀",
        "🎵", "🎶", "💦", "💭", "💣",
        "🎯", "🎨", "🎭", "🎪", "🎰",
        
        # Food & Drink
        "🍷", "🍺", "🍻", "☕", "🍵",
        
        # Symbols
        "💯", "✅", "❌", "⭕", "❗",
        "❓", "‼️", "⁉️", "💤", "💢"
    ]
    return COMMON_EMOJIS

CHAR_SUBSTITUTION_MAP = {
    'A': '@',    'a': '@',
    'B': '8',    'b': '6',
    'C': '(',    'c': '©',
    'D': ')',    'd': 'ð',
    'E': '3',    'e': '€',
    'F': 'ƒ',    'f': 'ƒ',
    'G': '6',    'g': '9',
    'H': '#',    'h': '♓',
    'I': '1',    'i': '!',
    'J': '7',    'j': 'ʝ',
    'K': '|<',   'k': '|¢',
    'L': '£',    'l': '|',
    'M': 'M',    'm': 'ᴍ',
    'N': 'И',    'n': 'ñ',
    'O': '0',    'o': '°',
    'P': '?',    'p': 'ρ',
    'Q': '2',    'q': 'φ',
    'R': '®',    'r': 'Я',
    'S': '5',    's': '§',
    'T': '7',    't': '†',
    'U': 'µ',    'u': 'υ',
    'V': '√',    'v': 'ν',
    'W': 'Ш',    'w': 'ω',
    'X': '×',    'x': '⨯',
    'Y': '¥',    'y': 'ч',
    'Z': '2',    'z': 'ℤ',
    '0': 'O',
    '1': 'l',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'g',
    '.': '·',
    ',': '`',
    ':': ';',
    ';': ':',
    '!': '¡',
    '?': '¿',
    '"': "''",
    "'": '`',
    '(': '{',
    ')': '}',
    '[': '(',
    ']': ')',
    '{': '[',
    '}': ']',
    '<': '‹',
    '>': '›',
    '/': '\\',
    '\\': '/',
    '|': 'ǀ',
    '+': '†',
    '-': '−',
    '*': '×',
    '=': '≡',
    '%': '‰',
    '$': '€',
    '&': '＆',
    '@': 'α',
    '#': '♯',
    '^': 'ˆ',
    '~': '∼',
    '`': '´'
}

KEYBOARD_NEIGHBOR_MAP = {
    'a': 'qwsxzy', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxcw', 'e': 'wrsdf34',
    'f': 'rtgvcd', 'g': 'tyuhbvf', 'h': 'yuijknbg', 'i': 'uojkl89', 'j': 'uikmnh',
    'k': 'iolmj', 'l': 'opk;', 'm': 'njk,', 'n': 'bhjm', 'o': 'iklp90',
    'p': 'ol;[0-', 'q': 'wa12', 'r': 'etdf45', 's': 'wedxzay', 't': 'ryfg56',
    'u': 'yhji78', 'v': 'cfgb', 'w': 'qase23', 'x': 'yzsdc', 'y': 'tghuj567xsaz',
    'z': 'asxytghuj765',
    '1': '2q`', '2': '13qw', '3': '24we', '4': '35er', '5': '46rt',
    '6': '57ty', '7': '68yu', '8': '79ui', '9': '80io', '0': '9-op',
    '`': '1', '-': '0=p', '=': '-[',
    '[': ']p', ']': '[\\', '\\': ']',
    ';': "l'", "'": ';',
    ',': 'm.', '.': ',/', '/': '.',
    ' ': 'zxcvbnm',  # Spacebar
    
    # Including shift-accessible characters
    '!': '@1', '@': '#!2', '#': '$@3', '$': '%#4', '%': '^$5',
    '^': '&%6', '&': '*^7', '*': '(&8', '(': ')*9', ')': '_)0',
    '_': '+_-', '+': '=',
    '{': '}[', '}': '{]|', '|': '}\\',
    ':': '"', '"': ':',
    '<': '>,', '>': '<?', '?': '>/'
}

BASIC_STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'can\'t', 'cannot', 'could', 'couldn\'t',
    'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each',
    'few', 'for', 'from', 'further',
    'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s',
    'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
    'let\'s',
    'me', 'more', 'most', 'mustn\'t', 'my', 'myself',
    'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such',
    'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up',
    'very',
    'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t',
    'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
    # Additional common words often considered as stop words
    'able', 'across', 'almost', 'always', 'among', 'another', 'anybody', 'anyone', 'anything', 'anywhere', 'around',
    'became', 'become', 'becomes', 'becoming', 'behind', 'beside', 'besides', 'beyond',
    'cannot', 'certain', 'certainly', 'come', 'comes', 'contain', 'containing', 'contains',
    'done', 'due', 'during',
    'either', 'else', 'elsewhere', 'enough', 'especially', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except',
    'first', 'followed', 'following', 'follows', 'front', 'full', 'further',
    'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'gone', 'got', 'gotten',
    'happen', 'happens', 'hardly', 'has', 'have', 'having', 'hence', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'however',
    'immediate', 'immediately', 'important', 'indeed', 'instead', 'into', 'inward',
    'just',
    'keep', 'keeps', 'kept', 'know', 'known', 'knows',
    'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks',
    'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'mean', 'meanwhile', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must',
    'name', 'namely', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'nobody', 'none', 'noone', 'normally', 'nothing', 'now', 'nowhere',
    'obviously', 'often', 'okay', 'once', 'one', 'ones', 'onto', 'other', 'others', 'otherwise', 'ought',
    'perhaps', 'possible', 'probably',
    'quite',
    'rather', 'really', 'right',
    'said', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'since', 'six', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'sure',
    'take', 'taken', 'tell', 'tends', 'th', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two',
    'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'upon', 'use', 'used', 'useful', 'uses', 'using', 'usually',
    'value', 'various', 'very', 'via',
    'viz', 'vs',
    'want', 'wants', 'way', 'well', 'went', 'whatever', 'whence', 'whenever', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'whither', 'whoever', 'whole', 'whose', 'why', 'will', 'willing', 'wish', 'within', 'without', 'wonder', 'would',
    'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
])

PHRASE_STOP_WORDS = set([
    # Common phrases and expressions
    'in order to', 'as well as', 'as long as', 'in addition to', 'with regard to',
    'in terms of', 'on the other hand', 'in fact', 'as a result', 'for example',
    'in general', 'in particular', 'such as', 'rather than', 'as far as',
    'in case', 'in spite of', 'with respect to', 'due to the fact that', 'for the purpose of',
    
    # Internet and technology related
    'http', 'https', 'www', 'com', 'org', 'net', 'edu', 'gov',
    'email', 'website', 'webpage', 'online', 'offline', 'internet', 'web',
    'download', 'upload', 'login', 'logout', 'username', 'password',
    
    # Miscellaneous
    'etc', 'i.e', 'e.g', 'vs', 'versus', 'viz', 'namely', 'item', 'things',
    'stuff', 'aspect', 'feature', 'characteristic', 'quality', 'quantity',
    'approximately', 'roughly', 'about', 'around', 'circa',
    'actually', 'literally', 'basically', 'essentially', 'virtually',
    'presumably', 'supposedly', 'allegedly', 'apparently', 'seemingly'
])

COMPLETE_STOP_WORDS = BASIC_STOP_WORDS.union(PHRASE_STOP_WORDS)