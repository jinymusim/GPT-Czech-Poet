class StropheParams:
    

    # Most Common Rhyme Schemas (Every Rhyme schema with presence over 0.36 %)
    RHYME_SCHEMES = ['ABAB', 'XXXX',
                 'XAXA','AABB', 
                 'XXXXXX','ABBA', 
                 'AAXX', 'AABBCC',
                 'ABABCC','ABABXX',
                 'AABCCB','XXAA', 
                 'XAAX', 'AXAX', 
                 'XAXAXX','XXABAB',
                 'ABBACC','AXAA', 
                 'XAABBX','AABCBC',
                 'AABBXX','ABBAXX',
                 'ABABAB','AAXA', 
                 'AXXA','XAXABB',
                 'XXAABB','XXAAXX',
                 'ABABAX','XXABBA',
                 'AAXBBX','XXXAXA',
                 'AAAX','XABABX',
                 'XABBAX','AAXXBB',
                 'AXABBX','ABABBX',
                 'XAAXBB','AAAA',
                 'XAAA','XAABXB',
                 'AXABXB','AXAXBB',
                  None]
    
    RHYME = RHYME_SCHEMES


    NORMAL_SCHEMES = ["ABAB", "ABBA", "AABB", "AABBCC", "ABABCC", "ABBACC", "ABBAAB"]

    # First 200 Most common endings
    VERSE_ENDS = ['ní', 'la', 'je', 'tí', 'ce', 'ti', 'ky', 'ku', 'li', 'jí', 'ně', 'né', 'vá', 'se', 'ny', 'ly', 'na', 'ne', 'nou', 
              'lo', 'ci', 'mi', 'ný', 'sti', 'ka', 'le', 'cí', 'ná', 'ží', 'čí', 'ho', 'dí', 'ší', 'du', 'lí', 'dy', 'nu', 'ří', 
              'ji', 'ru', 'tě', 'ře', 'stí', 'vy', 'ká', 'še', 'dá', 'ni', 'te', 'ví', 'mu', 'tu', 'ta', 'vé', 'val', 'va', 'lý', 
              'tá', 'že', 'ty', 'no', 'vu', 'lá', 'kem', 'chu', 'ků', 'bě', 'vý', 'sy', 'me', 'zí', 'hu', 'vě', 'lu', 'da', 'ry', 
              'rá', 'lé', 'ko', 'ři', 'de', 'hy', 'lem', 'tem', 'kou', 'vou', 'ši', 'há', 'sí', 'ze', 'be', 'ra', 'má', 'to', 'by', 
              'mě', 'su', 'té', 'si', 'ných', 'den', 'či', 'ký', 'ním', 'če', 'tý', 'ma', 'my', 'sem', 'nem', 'dě', 'ha', 'vat', 'ným', 
              'dem', 'dou', 'sta', 'dla', 'svět', 'zem', 'jen', 'dal', 'mí', 'hou', 'zas', 'sen', 'rem', 'nů', 'bu', 'e', 'ba', 'ké', 
              'til', 'jest', 'ství', 'děl', 'květ', 'tů', 'chem', 'lou', 'sám', 'bí', 'tou', 'dé', 'šel', 'nul', 'chá', 'vem', 'sa', 
              'hlas', 'pí', 'čas', 'dil', 'let', 'cích', 'lů', 'žil', 'mů', 'dál', 'cha', 'byl', 'nost', 'ček', 'zy', 'hý', 'nám', 'di', 
              'bou', 'tím', 'ži', 'tek', 'vil', 'jsem', 'sů', 'dech', 'men', 'tla', 'sá', 'zrak', 'chy', 'vám', 'vi', 'dý', 'rád', 'svou', 
              'ném', 've', 'py', 'vo', 'vým', 'nek', 'již', 'víc', 'kal', 'mé', 'dů', 'stá', 'dnes', 'sty', 'ven', None]
    ENDS = VERSE_ENDS
    # Years to bucket to
    POET_YEARS_BUCKETS = [1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, None]
    POET_YEARS = POET_YEARS_BUCKETS
    YEAR = POET_YEARS_BUCKETS
    # Possible Meter Types
    METER_TYPES = ["J","T","D","A","X","Y","N","H","P", None]
    METER = METER_TYPES
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
    
    # Basic Characters to consider in rhyme and syllables (43)
    VALID_CHARS = [""," ",'a','á','b','c','č','d','ď','e','é','ě',
               'f','g','h','i','í','j','k','l','m','n','ň',
               'o','ó','p','q','r','ř','s','š','t','ť','u',
               'ú','ů','v','w','x','y','ý','z','ž']
    CHARS = VALID_CHARS
    
    POEM_TYPES = ['mysticismus', 'dětství', 'bůh', 'láska', 'život', 'umění', 'poezie',
    'smutek', 'zoufalství', 'deprese', 'smrt', 'náboženství', 'příroda', 'krása', 'stárnutí', 'touha',
    'cestování', 'sny', 'zrození', 'válka', 'neúspěch', 'nesmrtelnost', 'fantazie']
    TYPES = POEM_TYPES
class Tokens:
# Tokenizers Special Tokens
    EOS = "<|EOS|>"
    EOS_ID = 0
    PAD = "<|PAD|>"
    PAD_ID = 1
    UNK = "<|UNK|>"
    UNK_ID = 2
    CLS = "<|CLS|>"
    CLS_ID = 3
    # SEP Token is EOS Token
    SEP = EOS
    SEP_ID = 0
    
    ALL_TOKENS = {
        EOS : 0,
        PAD : 1,
        UNK : 2,
        CLS : 3,
    }
    
    AUTHOR = "<|AUTHOR|>"
    CATEGORY = "<|CATEGORY|>"
    SUMMARY = "<|SUMMARY|>"
    TITLE = "<|TITLE|>"
    YEAR = "<|YEAR|>"
    STROPHE_START = "<|STROPHE_START|>"
    STROPHE_END = "<|STROPHE_END|>"
    
    METER = "<|METER|>"
    RHYME = "<|RHYME|>"


DEFAULT_STROPHE="""
<|STROPHE_START|>
<|METER|> J
<|RHYME|> ABAB
rád # a kdo by nedlel tak prudce rád,
spat # nežli se vytratí spat?
víc # nic nechci víc
víc # než nic a víc,
<|STROPHE_END|>
"""

import re
import numpy as np

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False

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
            year_index = np.argmin(np.abs(np.asarray(StropheParams.YEAR[:-1]) - int(raw_year)))
            return str(StropheParams.YEAR[year_index])
        else:
            return "NaN"
        
    _RHYME_POS = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
    @staticmethod
    def rhyme_sec(rhyme_ref, current_rhyme):
        """Return proper rhyme indicator to given reference
        
        Args:
            rhyme_ref (_type_): reference number of 'A'
            current_rhyme (_type_): current rhyme number that needs inidcation
            
        Returns:
            str: rhyme indicator character
        """
        
        return "X" if current_rhyme == None or current_rhyme== -1 or rhyme_ref == None or current_rhyme < rhyme_ref or current_rhyme >= rhyme_ref + len(TextManipulation._RHYME_POS) else TextManipulation._RHYME_POS[current_rhyme - rhyme_ref]
        
    @staticmethod
    def __post_process_rhyme(rhyme_str: str):
        # First Pass
        marker_count = {marker: rhyme_str.count(marker) for marker in TextManipulation._RHYME_POS}
        for key, val in marker_count.items():
            # Replace all, that ocurr only once with X
            if val == 1:
                rhyme_str = re.sub(key, 'X', rhyme_str)
        # Downscale higher to lower if lower not present
        marker_count = {marker: rhyme_str.count(marker) for marker in TextManipulation._RHYME_POS}
        for key, val in marker_count.items():
            if val > 1 and key != 'X':
                key_index = TextManipulation._RHYME_POS.index(key)
                replacements = {marker: rhyme_str.count(marker) for marker in TextManipulation._RHYME_POS[:key_index]}
                for rep_key, rep_val in replacements.items():
                    if rep_val ==0:
                        rhyme_str = re.sub(key, rep_key, rhyme_str)
                        break
                    
        # Pass to swap letters
        marker_index = {marker: rhyme_str.find(marker) for marker in TextManipulation._RHYME_POS if rhyme_str.find(marker) != -1}
        keys_values = marker_index.items()
        keys = [v[0] for v in keys_values]
        values = [v[1] for v in keys_values]
       
        i = 0
        while i < len(keys):
            j= 0
            while j< len(keys):
                if TextManipulation._RHYME_POS.index(keys[j]) > TextManipulation._RHYME_POS.index(keys[i]) and values[j] < values[i]:
                    # Swap the positions
                    rhyme_str = re.sub(keys[j], 'Z', rhyme_str)
                    rhyme_str = re.sub(keys[i], keys[j], rhyme_str)
                    rhyme_str = re.sub('Z', keys[i], rhyme_str)
                    # Need to update the value
                    temp = values[i]
                    values[i]= values[j]
                    values[j] = temp
                j+=1
            i+=1
        
            
        return rhyme_str
            
        
    @staticmethod
    def _rhyme_string(curr_rhyme_list):
        """Translate rhyme as list of rhyming number to rhyme schema

        Args:
            curr_rhyme_list (list): Current rhyme as list of ints indicating rhyming verses

        Returns:
            str: Rhyme schema
        """
        rhyme_list = curr_rhyme_list.copy()
        reference = None
        # Give None a blank -1 rhyme id
        for i in range(len(rhyme_list)):
            if rhyme_list[i] != None and reference == None:
                reference = rhyme_list[i]
            elif rhyme_list[i] != None and rhyme_list[i] < reference:
                reference = rhyme_list[i]
            elif rhyme_list[i] == None:
                 rhyme_list[i] = -1
        
        # With more robust post processing, this is may not needed
               
        # if there is valid rhyme, normalize 
        if reference != None:
            # sort the rhyme and get index of reference number
            cheat_sheet =  sorted(list(set(rhyme_list[:])))
            ref_index = cheat_sheet.index(reference)
            # normalize the rest around this reference
            for i in range(len(rhyme_list)):
                idx = cheat_sheet.index(rhyme_list[i])
                rhyme_list[i] = reference + (idx - ref_index)
        
                
        rhyme_str = ""
        for num in rhyme_list:
           rhyme_str += TextManipulation.rhyme_sec(reference, num)
        
        return TextManipulation.__post_process_rhyme(rhyme_str)

    _SHORTIFY_DICT = {"á": "a", "é": "e", "í":"i", 
                      "ů":"u", "ú":"u", "ó": "o", "ý": "y"}
    @staticmethod
    def _shortify(text: str):
        for key, value in TextManipulation._SHORTIFY_DICT.items():
            text = text.replace(key, value)
        return text

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
        return meter in StropheParams.METER[:-1]
    
    @staticmethod
    def _is_year(year:str):
        """Return if string is year or special NaN

        Args:
            year (str): string to analyze

        Returns:
            bool: If string is year or special NaN
        """
        return (year.isdecimal() and int(year) > 1_000 and int(year) < 10_000) or year == "NaN"
    
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
        
        rhyme_vec = np.zeros(len(StropheParams.RHYME))
        if rhyme in StropheParams.RHYME:
            rhyme_vec[StropheParams.RHYME.index(rhyme)] = 1
        else:
            rhyme_vec[-1] = 1
            
        return rhyme_vec
    
    
    @staticmethod
    def _publish_year_vector(year_string):
        """Construct vector of year of publishing, weighting by distance

        Args:
            year_string (str): String with publish year

        Returns:
            numpy.ndarray: Vector of bucketized One-hot encoded publish year
        """
        publish_year = None if not year_string.isdigit() else int(year_string)
        publish_vector = np.zeros(len(StropheParams.YEAR))
        if publish_year == None:
            publish_vector[-1] = 1
        else:
            # Distance Part
            #distance_weighting = [1/(1 + abs(year - publish_year)) for year in POET_YEARS_BUCKETS[:-1]] + [0]
            #publish_vector = np.asarray(distance_weighting)
            # Correct class correction
            publish_vector[np.argmin( abs(np.asarray(StropheParams.YEAR[:-1]) - publish_year))] += 1
            # Normalize
            #publish_vector = publish_vector/np.sum(publish_vector)
        return publish_vector  
    
    @staticmethod
    def _rhyme_or_not(rhyme_str:str) -> np.ndarray:
        """Create vector if given rhyme string is in our list of rhyme schemas

        Args:
            rhyme_str (str): string to construct vector from 

        Returns:
            numpy.ndarray: Boolean flag vector
        """
        rhyme_vector = np.zeros(2)
        if rhyme_str in StropheParams.RHYME:
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
        metre_vec = np.zeros(len(StropheParams.METER))
        if metre in StropheParams.METER:
            metre_vec[StropheParams.METER.index(metre)] = 1
        else:
            metre_vec[-1] = 1            
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
            if TextAnalysis._is_year(param):
                # Year is Bucketized so to fit
                poet_params["YEAR"] = TextManipulation._year_bucketor(param) 
            elif TextAnalysis._rhyme_like(param):
                poet_params["RHYME"] = param
            elif TextAnalysis._is_meter(param):
                poet_params["STROPHE_METER"] = param
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
        return end.isalpha() and end.islower() and len(end) <= 5
    
    @staticmethod
    def _continuos_line_analysis(text:str):
        """Analysis of Content lines for LENGTH, TRUE_LENGTH, END, TRUE_END

        Args:
            text (str): content line to analyze

        Returns:
            dict: Dictionary with analysis result
        """
        # Strip line of most separators and look if its empty
        line_striped = TextManipulation._remove_most_nonchar(text, lower_case=False).strip()
        if not line_striped:
            return {}
        line_params = {}
        # OLD MODEL
        if text.count('#') == 0: # BASIC
            pass
        else:
            for param_group in text.split('#')[:-1]:
                for param in param_group.split():
                    if TextAnalysis._is_meter(param.strip()):
                        line_params["METER"] = param.strip()
                    elif TextAnalysis._is_line_length(param.strip()):
                        line_params["LENGTH"] = int(param.strip())
                    elif TextAnalysis._is_line_end(param.strip()):
                        line_params["END"] = param.strip()
                    
                        
        line_params["TRUE_LENGTH"] = sum(map(len, SyllableMaker.syllabify(line_striped.split('#')[-1]) )) 
        line_only_char = TextManipulation._remove_all_nonchar(line_striped).strip()
        if len(line_only_char) > 2:
            line_params["TRUE_END"] = SyllableMaker.syllabify(" ".join(line_only_char.split()[-2:]))[-1][-1]
        
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
        return  "RHYME" in small_analysis.keys() or "YEAR" in small_analysis.keys()

    _CONSONANTS = ['b', 'c', 'č', 'd', 'ď', 'f', 'g', 'h', 'j', 'k', 
                   'l', 'm', 'n', 'ň', 'p', 'r', 'ř', 's', 'š', 't', 
                   'ť', 'v', 'z', 'ž']
    @staticmethod
    def is_consonant(char:str):
        return char in TextAnalysis._CONSONANTS
    
    _VOWELS = ['a', 'e', 'i', 'u', 'o', 'y', 'á', 
               "é", "í", "ú", "ů", "ó" ,"ý" ]
    @staticmethod
    def is_vowel(char:str):
        return char in TextAnalysis._VOWELS
    
    
    
    @staticmethod
    def _end_matches(verse:str, end:str):
        end_short = TextManipulation._shortify(end)
        verse_clean = TextManipulation._remove_all_nonchar(verse)
        if len(end_short)  > len(verse_clean):
            return False
        verse_end = TextManipulation._shortify(verse_clean[-len(end_short):])
        return end_short == verse_end
    
                
class SyllableMaker:
    """Static class with methods for separating string to list of Syllables

    Returns:
        list: List of syllables
    """
    
        
# NON-Original code!
# Taken from Barbora Štěpánková

    @staticmethod
    def syllabify(text : str) -> list[list[str]]:
        words = re.findall(r"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+", text)
        syllables : list[list[str]] = []

        i = 0
        while i < len(words):
            word = words[i]

            if (word.lower() == "k" or word.lower() == "v" or word.lower() == "s" or word.lower() == "z") and i < len(words) - 1 and len(words[i + 1]) > 1:
                i += 1
                word = word + words[i]

            letter_counter = 0
            _syllable = []
            # Get syllables: mask the word and split the mask
            for syllable_mask in SyllableMaker.__split_mask(SyllableMaker.__create_word_mask(word)):
                word_syllable = ""
                for character in syllable_mask:
                    word_syllable += word[letter_counter]
                    letter_counter += 1
                _syllable.append(word_syllable)

            i += 1
            
            syllables.append(_syllable)

        return list(filter(lambda x: len(x) > 0, syllables))


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

class VersologicalMaker:
    """Static class with methods for separating string to list of Versological Units (V K)

    Returns:
        list: List of Units
    """
    

    @staticmethod
    def verse_segmnent(text : str) -> list[list[str]]:
        words = re.findall(r"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+", text)
        vers_units : list[list[str]] = []

        i = 0
        while i < len(words):
            word = words[i]

            letter_counter = 0
            _syllable = []
            # Get syllables: mask the word and split the mask
            for syllable_mask in VersologicalMaker.__split_mask(VersologicalMaker.__create_word_mask(word)):
                word_syllable = ""
                for character in syllable_mask:
                    word_syllable += word[letter_counter]
                    letter_counter += 1
                _syllable.append(word_syllable)

            i += 1
            
            vers_units.append(_syllable)

        return list(filter(lambda x: len(x) > 0, vers_units))


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
    
            # now all vocals
    		(vocals, 'V'),

            # r,l that act like vocals in syllables
    		(r'([^V])([rl])(0*[^0Vrl]|$)', r'\1V\3'),


    		(consonants, 'K')
    	]

        for (original, replacement) in replacements:
            word = re.sub(original, replacement, word)

        return word


    @staticmethod
    def __split_mask(mask : str) -> list[str]:
        replacements = [
    		# vocal at the beginning
            (r'(^(K0*)+)', r'\1/'),

    		# dividing the middle of the word
    		(r'((V0*)+(K0*)+)', r'\1/'),
            (r'/((V0*)+)$', r'/\1'),

    		# add the last consonant to the previous syllable
    		#(r'/(K0*)$', r'\1/')
    	]

        for (original, replacement) in replacements:
            mask = re.sub(original, replacement, mask)

        if len(mask) > 0 and mask[-1] == "/":
            mask = mask[0:-1]

        return mask.split("/")

