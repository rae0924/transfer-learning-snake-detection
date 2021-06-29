from bing_image_collector.collector import BingImageCollector


api_key = 'ad91c53c6ed84d549d0e2a24d1d39e50'

# some classes of snake in the imagenet competition
snake_terms = [
    'worm snake',
    'ringneck snake',
    'hognose snake', 
    'puff adder', 
    'sand viper',
    'green snake',
    'grass snake',
    'king snake',
    'garter snake',
    'water snake',
    'vine snake',
    'night snake',
    'boa constrictor',
    'rock python',
    'indian cobra',
    'green mamba',
    'sea snake'
    'horned viper',
    'diamondback rattlesnake',
]

# random picked set of animals from imagenet
other_terms = [
    'great white shark',
    'house finch',
    'magpie',
    'vulture',
    'spotted salamander',
    'bullfrog',
    'box turtle',
    'frilled lizard',
    'komodo dragon',
    'nile crocodile',
    'scorpion',
    'tarantula',
    'centipede',
    'peacock',
    'lorikeet',
    'toucan',
    'goose',
    'platypus',
    'wombat',
    'jellyfish',
    'slug',
    'rock crab',
    'crayfish',
    'flamingo',
    'bustard',
    'pelican',
    'grey whale',
    'sea lion',
    'dog',
    'grey wolf',
    'dingo',
    'hyena',
    'arctic fox',
    'cat',
    'cougar',
    'jaguar',
    'brown bear',
    'meerkat',
    'dung beetle',
    'grasshopper',
    'mantis',
    'dragonfly',
    'monarch',
    'starfish',
    'hare',
    'marmot',
    'zebra',
    'wild boar',
    'hippopotamus',
    'bison',
    'ibex',
    'llama',
    'otter',
    'sloth',
    'chimpanzee',
    'baboon',
    'marmoset',
    'spider monkey',
    'african elephant',
    'eel',
    'sturgeon',
    'pufferfish'
]

if __name__ == "__main__":
    collector = BingImageCollector()
    collector.collect(
        search_terms=snake_terms,
        api_key=api_key,
        num=100,
        root_dir='./data/snake',
        workers=4,
    )
    collector.collect(
        search_terms=other_terms,
        api_key=api_key,
        num=30,
        root_dir='./data/non-snake',
        workers=4,
    )

    
