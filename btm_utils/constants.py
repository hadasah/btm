TRAIN_HYPERPARAMS = {
    "125M": {
        'NUM_GPUS': 16,
        "NUM_BRANCH_GPUS": 2,
        "NUM_STEPS": 80000,
        "TRAIN_HOURS": -1,
        "SEED_PHASE": 40000,
        "SAVE_INTERVAL_UPDATES": 8000,
        "LR": 5e-4,
        "UF": 32
    },
    "350M": {
        "NUM_GPUS": 32,
        "NUM_BRANCH_GPUS": 4,
        "NUM_STEPS": 32000,
        "SEED_PHASE": 16000,
        "TRAIN_HOURS": -1,
        "SAVE_INTERVAL_UPDATES": 4000,
        "LR": 5e-4,
        "UF": 32
    },
    "750M": {
        'NUM_GPUS': 64,
        "NUM_BRANCH_GPUS": 8,
        "NUM_STEPS": 24000,
        "TRAIN_HOURS": -1,
        "SEED_PHASE": 12000,
        "SAVE_INTERVAL_UPDATES": 2000,
        "LR": 5e-4,
        "UF": 32
    },
    "1.3B": {
        'NUM_GPUS': 128,
        "NUM_BRANCH_GPUS": 16,
        "NUM_STEPS": 12000,
        "TRAIN_HOURS": -1,
        "SEED_PHASE": 6000,
        "SAVE_INTERVAL_UPDATES": 2000,
        "LR": 5e-4,
        "UF": 32
    },
}

TRAIN_DOMAINS = [
    '1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews',
    'cs', 'legal', 'med', 'reddit']

EVAL_DOMAINS = ['anonymized_latest_news_redo', 'anonymized_tweets_redo', 
    'anonymized_yelp_reviews_redo', 'cord19-redo', 'github_redo', 'gutenberg', 
    'legal_contracts', 'qasper']

SCALED_DOMAINS = [
    '2021newscrawl',
    'Biology',
    'Books',
    'Business',
    'C#',
    'C++',
    'CSS',
    'C',
    'Chemistry',
    'ClothingShoesandJewelry',
    'Economics',
    'Electronics',
    'Engineering',
    'GO',
    'Geography',
    'Geology',
    'HTML',
    'History',
    'HomeandKitchen',
    'JavaScript',
    'Java',
    'Markdown',
    'Mathematics',
    'MoviesandTV',
    'PHP',
    'Philosophy',
    'Physics',
    'Psychology',
    'Python',
    'Ruby',
    'Sociology',
    'SportsandOutdoors',
    'acl',
    'anonymizedyelpreviewsredodemixpaper',
    'art',
    'bookcorpus',
    'c4',
    'codecontests',
    'cord19',
    'dmmathematics',
    'envsci',
    'gamingcomments',
    'gutenberg',
    'hackernews',
    'materials',
    'na',
    'opensubtitles',
    'polisci',
    'sportscomments',
    'stackexchange',
    'stackoverflow',
    'stories',
    'supremecourt',
    'twitter',
    'uspto',
    'wikipedia',
]