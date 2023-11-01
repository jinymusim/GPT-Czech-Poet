# Most Common Rhyme Schemas
RHYME_SCHEMES = ["ABAB", "ABBA", 
                 "XAXA", "ABCB", 
                 "AABB", "AABA", 
                 "AAAA", "AABC",
                 'XXXX', 'AXAX',
                 "AABBCC", "AABCCB", 
                 "ABABCC", 'AABCBC', 
                 "AAABAB", "ABABXX" 
                 "ABABCD", "ABABAB", 
                 "ABABBC", "ABABCB",
                 "ABBAAB","AABABB",
                 "ABCBBB",'ABCBCD', 
                 "ABBACC","AABBCD",
                  None]

NORMAL_SCHEMES = ["ABAB", "ABBA", "AABB", "AABBCC", "ABABCC", "ABBACC", "ABBAAB"]

# First 200 Most common endings
VERSE_ENDS = ['ní', 'ou', 'em', 'la', 'ch', 'ti', 'tí', 'je', 'li', 'al', 'ce', 'ky', 'ku', 'ně', 'jí', 'ly', 'il', 'en', 'né', 
                'lo', 'ne', 'vá', 'ny', 'se', 'na', 'ím', 'st', 'le', 'ný', 'ci', 'mi', 'ka', 'ná', 'lí', 'cí', 'ží', 'čí', 'ám', 
                'hu', 'ho', 'ří', 'dí', 'nu', 'dy', 'ší', 'ví', 'du', 'ta', 'as', 'tě', 'ře', 'ru', 'vé', 'ým', 'at', 'ek', 'el', 
                'te', 'tu', 'ká', 'ji', 'ět', 'ni', 'še', 'vy', 'dá', 'it', 'tá', 'ty', 'lý', 'lá', 'mu', 'va', 'ém', 'ěl', 'no', 
                'že', 'vu', 'ál', 'há', 'ků', 'vý', 'bě', 'hy', 'lé', 'sy', 'me', 'es', 'ra', 'ak', 'ad', 'ry', 'zí', 'et', 'rá', 
                'de', 'vě', 'ři', 'lu', 'át', 'da', 'ko', 'ha', 'té', 'to', 'ed', 'ít', 'ký', 'ši', 'íš', 'sí', 'íc', 'ze', 'si', 
                'be', 'má', 'mě', 'by', 'su', 'tý', 'ej', 'či', 'če', 'my', 'ké', 'án', 'ma', 'ům', 'or', 'nů', 'áš', 'dě', 'ec', 
                'mí', 'ev', 'ád', 'ut', 'am', 'yl', 'ul', 'tů', 'bu', 'ás', 'ba', 'ud', 'ář', 'ie', 'od', 'pí', 'ůj', 'eš', 'hý', 
                'bí', 'íž', 'dé', 'an', 'sa', 've', 'lů', 'ín', 'id', 'in', 'mů', 'di', 'hů', 'ic', 'on', 'eň', 'zy', 'ol', 'vo', 
                'ži', 'sů', 'ík', 'vi', 'oj', 'uk', 'uh', 'oc', 'iž', 'sá', 'ěv', 'dý', 'av', 'iv', 'rů', 'ot', 'py', 'mé', 'um', 
                'zd', 'dů', 'ar', 'rý', 'aň', 'sk', 'ok', 'om', 'už', 'ěk', 'ov', 'er', 'uď', 'bi', 'áz', 'ýt', 'ěm', 'ik', 'eď', 
                'ob', 'ák', 'ůh', 'ár', 'sť', 'ro', 'yt', 'ěj', 'mý', 'us', 'ěn', 'ii', 'hé', 'áj', 'pá', 'íh', 'ih', 'zi', 'bá', 
                'eč', 'ré', 'ír', 'ců', 'uj', 'dl', 'áh', 'ův', 'aj', 'eh', 'éž', 'pu', 'ýš', 'zu', 'im', 're', 'up', 'os', 'ah', 
                'rt', 'mo', 'áň', 'sl', 'íl', 'cy', 'ys', 'hl', 'oh', 'ěz', 'ěs', 'ež', 'ií', 'vů', 'kl', 'az', 'cý', 'pe', 'ěd', 
                'do', 'yn', 'šť', 'ez', 'ůl', 'ub', 'ln', 'yk', 'pý', 'ěc', 'ať', 'já', 'op', 'eb', 'áč', 'ív', 'áv', 'jů', 'sý', 
                'is', ' a', 'iť', 'ěř', 'za', 'uť', 'ěh', 'pě', 'íp', 'áž', 'ěď', 'bů', 'ep', 'iš', 'yš', 'ia', 'pa', 'un', 'ěť', 
                'pů', 'eř', 'tr', 'nt', 'pi', 'tl', 'eť', 'ju', 'oď', 'řů', 'ýr', 'rh', 'ur', 'zý', 'ěž', 'ýn', 'ip', 'bý', 'pé', 
                'íň', 'zů', 'čů', 'uč', 'éb', 'ap', 'ón', 'uř', 'ůr', 'íř', 'ač', 'co', 'íč', 'až', 'ls', 'ůž', 'ěr', 'oč', 'ič', 
                'ař', 'ěš', 'uv', 'ůz', 'oň', 'bé', 'sé', 'yč', 'áť', 'jď', 'ri', 'íť', 'oš', 'ůň', 'ék', 'uc', 'rk', 'bo', 'ýl', 
                'oť', 'íz', 'lh', 'so', 'áb', 'ja', 'ij', 'ůn', 'rv', 'žů', 'ab', 'he', 'íd', 'ér', 'uš', 'ýž', 'fá', 'rs', 'rn', 
                'iz', 'ib', 'ki', 'éd', 'év', 'rd', 'yb', 'oz', 'oř', 'ét', 'ož', 'ga', 'yň', 'rp', 'nd', 'of', 'rť', 'iď', 'ýv', 
                'yz', None]
# Years to bucket to
POET_YEARS_BUCKETS = [1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, None]
# Possible Meter Types
METER_TYPES = ["J","T","D","A","X","Y","N","H","P", None]
# Translation of Meter to one char types
METER_TRANSLATE = {
    "J":"J",
    "T":"T",
    "D":"D",
    "A":"A",
    "X":"X",
    "Y":"Y",
    "hexameter": "H",
    "pentameter": "P",
    "N":"N"
}
# Tokenizers Special Tokens
PAD = "<|PAD|>"
UNK = "<|UNK|>"
EOS = "<|EOS|>"
# Basic Characters to consider in rhyme and syllables (43)
VALID_CHARS = [""," ",'a','á','b','c','č','d','ď','e','é','ě',
               'f','g','h','i','í','j','k','l','m','n','ň',
               'o','ó','p','q','r','ř','s','š','t','ť','u',
               'ú','ů','v','w','x','y','ý','z','ž']

import re
import numpy as np

class TextManipulation:
    """Static class for string manipulation methods

    Returns:
        _type_: str returned by all methods
    """
        
    @staticmethod
    def _remove_most_nonchar(raw_text, lower_case=True):
        """Remove most non-alpha non-whitespace characters

        Args:
            raw_text (str): Text to manipulate
            lower_case (bool, optional): If resulting text should be lowercase. Defaults to True.

        Returns:
            str: Cleaned up text
        """
        text = re.sub(r'[–\„\“\’\;\:()\]\[\_\*\‘\”\'\-\—\"]+', "", raw_text)
        return text.lower() if lower_case else text
    
    @staticmethod
    def _remove_all_nonchar(raw_text):
        """Remove all possible non-alpha characters

        Args:
            raw_text (str): Text to manipulate

        Returns:
            str: Cleaned up text
        """
        sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
        return sub
    
    @staticmethod
    def _year_bucketor(raw_year):
        """Bucketizes year string to boundaries, Bad inputs returns NaN string

        Args:
            raw_year (str): Year string to bucketize

        Returns:
            _type_: Bucketized year string
        """
        if TextAnalysis._is_year(raw_year) and raw_year != "NaN":
            year_index = np.argmin(np.abs(np.asarray(POET_YEARS_BUCKETS[:-1]) - int(raw_year)))
            return str(POET_YEARS_BUCKETS[year_index])
        else:
            return "NaN"

class TextAnalysis:
    """Static class with methods of analysis of strings

    Returns:
        Union[str, bool, dict, numpy.ndarray]: Analyzed input
    """
    
    # Possible Keys if returned type is dict
    POET_PARAM_LIST = ["RHYME", "YEAR", "METER", "LENGTH", "END", "TRUE_LENGTH", "TRUE_END"]
    
    @staticmethod
    def _is_meter(meter:str):
        """Return if string is meter type

        Args:
            meter (str): string to analyze

        Returns:
            bool: If string is meter type
        """
        return meter in METER_TYPES[:-1]
    
    @staticmethod
    def _is_year(year:str):
        """Return if string is year or special NaN

        Args:
            year (str): string to analyze

        Returns:
            bool: If string is year or special NaN
        """
        return (year.isdigit() and int(year) > 1_000 and int(year) < 10_000) or year == "NaN"
    
    @staticmethod
    def _rhyme_like(rhyme:str):
        """Return if string is structured like rhyme schema

        Args:
            rhyme (str): string to analyze

        Returns:
            bool: If string is structured like rhyme schema
        """
        return (rhyme.isupper() and len(rhyme) >= 3 and len(rhyme) <= 6)
    
    @staticmethod
    def _rhyme_vector(rhyme:str) -> np.ndarray:
        """Create One-hot encoded rhyme schema vector from given string

        Args:
            rhyme (str): string to construct vector from 

        Returns:
            numpy.ndarray: One-hot encoded rhyme schema vector
        """
        
        rhyme_vec = np.zeros(len(RHYME_SCHEMES))
        if rhyme in RHYME_SCHEMES:
            rhyme_vec[RHYME_SCHEMES.index(rhyme)] = 1
        else:
            rhyme_vec[-1] = 1
            
        return rhyme_vec
    
    @staticmethod
    def _rhyme_or_not(rhyme_str:str) -> np.ndarray:
        """Create vector if given rhyme string is in our list of rhyme schemas

        Args:
            rhyme_str (str): string to construct vector from 

        Returns:
            numpy.ndarray: Boolean flag vector
        """
        rhyme_vector = np.zeros(2)
        if rhyme_str in RHYME_SCHEMES:
            rhyme_vector[0] = 1
        else:
            rhyme_vector[1] = 1
        return rhyme_vector
        
    @staticmethod
    def _metre_vector(metre: str) -> np.ndarray:
        """Create One-hot encoded metre vector from given string

        Args:
            metre (str): string to construct vector from 

        Returns:
            numpy.ndarray: One-hot encoded metre vector
        """
        metre_vec = np.zeros(len(METER_TYPES))
        if metre in METER_TYPES:
            metre_vec[METER_TYPES.index(metre)] = 1
        else:
            metre_vec[-2] = 1            
        return metre_vec
    
    @staticmethod
    def _first_line_analysis(text:str):
        """Analysis of parameter line for RHYME, METER, YEAR

        Args:
            text (str): parameter line string

        Returns:
            dict: Dictionary with analysis result
        """
        line_striped = text.strip()
        if not line_striped:
            return {}
        poet_params = {}
        # Look for each possible parameter
        for param in line_striped.split():
            if TextAnalysis._is_meter(param):
                poet_params["METER"] = param
            elif TextAnalysis._is_year(param):
                # Year is Bucketized so to fit
                poet_params["YEAR"] = TextManipulation._year_bucketor(param) 
            elif TextAnalysis._rhyme_like(param):
                poet_params["RHYME"] = param
        return poet_params
    
    @staticmethod
    def _is_line_length(length:str):
        """Return if string is number of syllables parameter

        Args:
            length (str): string to analyze

        Returns:
            bool: If string is number of syllables parameter
        """
        return length.isdigit() and int(length) > 1 and int(length) < 100
    
    @staticmethod
    def _is_line_end(end:str):
        """Return if string is valid ending syllable/sequence parameter

        Args:
            end (str): string to analyze

        Returns:
            bool: If string is valid ending syllable/sequence parameter
        """
        return end.isalpha()  and len(end) <= 5 
    
    @staticmethod
    def _continuos_line_analysis(text:str):
        """Analysis of Content lines for LENGTH, TRUE_LENGTH, END, TRUE_END

        Args:
            text (str): content line to analyze

        Returns:
            dict: Dictionary with analysis result
        """
        # Strip line of most separators and look if its empty
        line_striped = TextManipulation._remove_most_nonchar(text).strip()
        if not line_striped:
            return {}
        line_params = {}
        # Look for parameters in Order LENGTH, END, TRUE_LENGTH, TRUE_END
        if TextAnalysis._is_line_length(line_striped.split()[0]):
            line_params["LENGTH"] = int(line_striped.split()[0])
        if len(line_striped.split()) > 1 and TextAnalysis._is_line_end(line_striped.split()[1]):
            line_params["END"] = line_striped.split()[1]        
        if len(line_striped.split()) > 3:
            line_params["TRUE_LENGTH"] = len(SyllableMaker.syllabify(" ".join(line_striped.split()[3:])))
        # TRUE_END needs only alpha chars, so all other chars are removed    
        line_only_char = TextManipulation._remove_all_nonchar(line_striped).strip()
        if len(line_only_char) > 2:
            line_params["TRUE_END"] = SyllableMaker.syllabify(line_only_char)[-1]
        
        return line_params
    
    @staticmethod
    def _is_param_line(text:str):
        """Return if line is a Parameter line (Parameters RHYME, METER, YEAR)

        Args:
            text (str): line to analyze

        Returns:
            bool: If line is a Parameter line
        """
        line_striped = text.strip()
        if not line_striped:
            return False
        small_analysis = TextAnalysis._first_line_analysis(line_striped)
        return  "RHYME" in small_analysis.keys() or "METER" in small_analysis.keys() or "YEAR" in small_analysis.keys()
    
# NON-Original code!
# Taken from Barbora Štěpánková
class SyllableMaker:
    """Static class with methods for separating string to list of Syllables

    Returns:
        list: List of syllables
    """

    @staticmethod
    def syllabify(text : str) -> list[str]:
        words = re.findall(r"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+", text)
        syllables : list[str] = []

        i = 0
        while i < len(words):
            word = words[i]

            if (word.lower() == "k" or word.lower() == "v" or word.lower() == "s" or word.lower() == "z") and i < len(words) - 1 and len(words[i + 1]) > 1:
                i += 1
                word = word + words[i]

            letter_counter = 0

            # Get syllables: mask the word and split the mask
            for syllable_mask in SyllableMaker.__split_mask(SyllableMaker.__create_word_mask(word)):
                word_syllable = ""
                for character in syllable_mask:
                    word_syllable += word[letter_counter]
                    letter_counter += 1

                syllables.append(word_syllable)

            i += 1

        return syllables


    @staticmethod
    def __create_word_mask(word : str) -> str:
        word = word.lower()

        vocals = r"[aeiyouáéěíýóůúäöü]"
        consonants = r"[bcčdďfghjklmnňpqrřsštťvwxzž]"

        replacements = [
            #double letters
    		('ch', 'c0'),
    		('rr', 'r0'),
            ('ll', 'l0'),
    		('nn', 'n0'),
    		('th', 't0'),

            # au, ou, ai, oi
    		(r'[ao]u', '0V'),
            (r'[ao]i','0V'),

            # eu at the beginning of the word
    		(r'^eu', '0V'),
    
            # now all vocals
    		(vocals, 'V'),

            # r,l that act like vocals in syllables
    		(r'([^V])([rl])(0*[^0Vrl]|$)', r'\1V\3'),

            # sp, st, sk, št, Cř, Cl, Cr, Cv
    		(r's[pt]', 's0'),
    		(r'([^V0lr]0*)[řlrv]', r'\g<1>0'),
    		(r'([^V0]0*)sk', r'\1s0'),
    		(r'([^V0]0*)št', r'\1š0'),

    		(consonants, 'K')
    	]

        for (original, replacement) in replacements:
            word = re.sub(original, replacement, word)

        return word


    @staticmethod
    def __split_mask(mask : str) -> list[str]:
        replacements = [
    		# vocal at the beginning
    		(r'(^0*V)(K0*V)', r'\1/\2'),
    		(r'(^0*V0*K0*)K', r'\1/K'),

    		# dividing the middle of the word
    		(r'(K0*V(K0*$)?)', r'\1/'),
    		(r'/(K0*)K', r'\1/K'),
    		(r'/(0*V)(0*K0*V)', r'/\1/\2'),
    		(r'/(0*V0*K0*)K', r'/\1/K'),

    		# add the last consonant to the previous syllable
    		(r'/(K0*)$', r'\1/')
    	]

        for (original, replacement) in replacements:
            mask = re.sub(original, replacement, mask)

        if len(mask) > 0 and mask[-1] == "/":
            mask = mask[0:-1]

        return mask.split("/")
