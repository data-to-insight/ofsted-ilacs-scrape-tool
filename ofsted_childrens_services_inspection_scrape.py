#
# Export options

export_summary_filename = 'ofsted_ilacs_overview'
# export_file_type         = 'csv' # Excel / csv currently supported
export_file_type         = 'excel'

# Default (sub)folder structure
# Defined to offer some ease of onward flexibility

# data exports
root_export_folder = 'export_data'              # <all> exports folder
inspections_subfolder = 'inspection_reports'    # downloaded report pdfs

# data imports
import_la_data_path = 'import_data/la_lookup/'
import_geo_data_path = 'import_data/geospatial/'
geo_boundaries_filename = 'local_authority_districts_boundaries.json'

# scrape inspection grade/data from pdf reports
pdf_data_capture = True # True is default (scrape within pdf inspection reports for inspection results etc)
                        # This impacts run time E.g False == ~1m20 / True == ~ 4m10
                        # False == only pdfs/list of LA's+link to most recent exported. Not inspection results.


repo_path = '/workspaces/ofsted-ilacs-scrape-tool'




#
# Ofsted site/page admin settings

short_inspection_threshold    = 7 # ILACS inspection duration (in days)
standard_inspection_threshold = 14

max_page_results = 200 # Set max number of search results to show on page(MUST be > total number of LA's!) 
url_stem = 'https://reports.ofsted.gov.uk/'


search_url = 'search?q=&location=&lat=&lon=&radius=&level_1_types=3&level_2_types%5B%5D=12'
max_page_results_url = '&rows=' + str(max_page_results) # Coerce results page to display ALL providers on single results page without next/pagination

# resultant complete url to process
url = url_stem + search_url + max_page_results_url 





# #
# # In progress Ofsted site/search link refactoring

# search_category = 3         # Default 3  == 'Childrens social care' (range 1 -> 4)
# search_sub_category = 12    # Default 12 == 'Local Authority Childrens Services' (range 8 -> 12)

# url_search_stem = 'search?q=&location=&radius='
# url = url_stem + url_search_stem + '&level_1_types=' + str(search_category) + '&level_2_types=' + str(search_sub_category) + max_page_results_url


#
# Script admin settings

# Non-standard modules that might need installing
import os
import io
import requests
from requests.exceptions import RequestException #  HTTP requests excep' class

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re       
from datetime import datetime, timedelta # timedelta enables server time adjustment
import json
import git # possible case for just: from git import Repo

import nltk
nltk.download('punkt')      # tokeniser models/sentence segmentation
nltk.download('stopwords')  # stop words ready for text analysis|NLP preprocessing
nltk.download('punkt_tab')  # added 120824 RH - as work-around fix textblob.exceptions.MissingCorpusError line 1384, in get_sentiment_and_topics

# #sentiment
# # nlp stuff for sentiment
# try:
#     from textblob import TextBlob
#     from gensim import corpora, models
#     # sh "/Applications/Python 3.11/Install Certificates.command"
# except ModuleNotFoundError:
#     print("Please install 'textblob' and 'gensim' using pip")


# pdf search/data extraction
try:
    import tabula  
    import PyPDF2   # continue use for now, but ..
    # import pypdf  # ...in PyPDF2 v3.0+, the correct import is now pypdf

reader = pypdf.PdfReader(buffer)  # Update usage
except ModuleNotFoundError:
    print("Please install 'tabula-py' and 'PyPDF2' using pip")


# handle optional excel export+active file links
try:
    import xlsxwriter
except ModuleNotFoundError:
    print("Please install 'openpyxl' and 'xlsxwriter' using pip")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
except ModuleNotFoundError:
    print("Please install 'scikit-learn' using pip")

# Configure logging/logging module
import warnings
import logging

# wipe / reset the logging file 
with open('output.log', 'w'):
    # comment out if maintaining ongoing/historic log
    pass

# Keep warnings quiet unless priority
logging.getLogger('org.apache.pdfbox').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')



#
# Function defs

def get_soup(url, retries=3, delay=5):
    """
    Given a URL, returns a BeautifulSoup object + request error handling
    Args:
        url (str):      The URL to fetch and parse
        retries (int):  Number of retries on network errors
        delay (int):    Delay between retries in seconds
    Returns:
        BeautifulSoup: The parsed HTML content, or None if an error occurs
    """
    timeout_seconds = 10  # lets not assume the Ofsted page is up, avoid over-pinging

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()  # any HTTP errors?
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except Timeout:
            print(f"Timeout getting URL '{url}' on attempt {attempt + 1}. Retrying after {delay} secs...")
            time.sleep(delay)
        except HTTPError as e:
            print(f"HTTP error getting URL '{url}': {e}")
            return None  # end retries on client and server errors
        except RequestException as e:
            print(f"Request error getting URL '{url}': {e}")
            if attempt < retries - 1:
                print(f"Retrying after {delay} secs...")
                time.sleep(delay) # pause to assist not getting blocked
            else:
                print("Max rtry attempts reached, giving up")
                return None
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return None

    return None  # All the retries failed / stop point


def clean_provider_name(name):
    """
    Cleans the la/provider name according to:
                - expected output based on existing ILACS sheet
                - historic string issues seen on Ofsted site

    Args:
        name (str): The original name to be cleaned.
    Returns:
        str: The cleaned name.
    """
    # Convert to lowercase and remove extra spaces
    name = name.lower().replace('  ', ' ')
    
    # Remove specific phrases
    name = name.replace("royal borough of ", "").replace("city of ", "").replace("metropolitan district council", "").replace("london borough of", "").replace("council of", "")

    # Remove further undesired 'single' words and join the remaining parts
    name_parts = [part for part in name.split() if part not in ['city', 'metropolitan', 'borough', 'council', 'county', 'district', 'the']]
    return ' '.join(name_parts)


def get_framework_type(start_date, end_date, short_inspection_threshold, standard_inspection_threshold):
    """
    Returns an inspection framework type based on the duration between the start and end dates.
    Dates are scraped, as this currently the only ref. This not ideal as based entirely
    on varied formats of text based data. Therefore some cleaning included here. 

    Args:
        start_date (str): Start date in the format "dd/mm/yyyy".
        end_date (str): End date in the format "dd/mm/yyyy".

    Returns:
        str: Inspection framework type, which can be "short", "standard", or "inspection duration longer than standard framework".
    """

    # Check if both start and end dates have been accessible
    if start_date is not None and end_date is not None:

        # Check if end date is not earlier than start date
        if end_date < start_date:
            inspection_framework_str = "invalid end or start date extracted"

        # Calculate the number of days between inspection start and end dates
        else:
            delta = end_date - start_date
            inspection_duration_days = delta.days

            # Determine the inspection framework based on the duration days
            # Note: Needs further investigation to sense check real-world timeframes here, i.e. are thresholds 'working days'?
            # For most instances this appears to be sufficiently accurate as-is. 
            if inspection_duration_days <= short_inspection_threshold:
                inspection_framework_str = "short"
            elif short_inspection_threshold < inspection_duration_days <= standard_inspection_threshold + 1:
                inspection_framework_str = "standard"
            else:
                inspection_framework_str = "inspection duration longer than standard framework"

    # Handle cases where start or end date is not provided 
    # Note: end date most likely to have not been extracted due to formatting issues
    else:
        inspection_framework_str = "invalid date format"

    return inspection_framework_str


def format_date(date_str: str, input_format: str, output_format: str) -> str:
    """
    Convert and format a date string.

    Args:
        date_str (str): The input date string.
        input_format (str): The format of the input date string.
        output_format (str): The desired output format.

    Returns:
        str: The formatted date string.
    """
    dt = datetime.strptime(date_str, input_format)
    date_obj = dt.date()

    return date_obj.strftime(output_format)


def parse_date(date_str, date_format):
    try:
        dt = datetime.strptime(date_str, date_format)
        return dt.date()  # only need date 
    except (TypeError, ValueError):
        return None
    

def format_date_for_report(date_obj, output_format_str):
    """
    Formats a datetime object as a string in the d/m/y format, or returns an empty string if the input is None.

    Args:
        date_obj (datetime.datetime or None): The datetime object to format, or None.

    Returns:
        str: The formatted date string, or an empty string if date_obj is None.
    """
    if date_obj is not None:
        return date_obj.strftime(output_format_str)
    else:
        return ""


def extract_inspection_grade(row, column_name):
    """
    Extracts the grade from the given row and column name. If the grade contains
    the phrase "requires improvement", it now returns the cleaned-up value.
    
    Args:
        row (pd.Series): A row from a Pandas DataFrame.
        column_name (str): The name of the column containing the grade.
    
    Returns:
        str: The extracted grade.
    
    Raises:
        ValueError: If the grade value cannot be converted to a string.
    """
    try:
        grade = str(row[column_name]).replace('\n', ' ').strip().lower()

        if "requires improvement" in grade:
            # Some RI text has further comment that we don't want, i.e. 'RI, *to become good*' 
            grade = "requires improvement"
        return grade
    except Exception as e:
        grade = f"Unknown value type : {grade}"
        error_msg = f"original error: {str(e)}, unknown value found: \"unknown : {grade}\""
        raise ValueError(error_msg)






def fix_invalid_judgement_table_structure(df):
    """
    Function to correct the structure of a given DataFrame that represents a judgement table. The function specifically 
    checks if the required headers ("judgement", "grade") are present in each row. If the headers are found, the 
    DataFrame is trimmed down to only columns containing the headers, non-header rows are removed and the correct 
    row is set as the new headers. If the headers are not found in any of the rows, a placeholder DataFrame 
    is returned instead.

    Parameters
    df : pandas.DataFrame
        The DataFrame containing the judgement table data to be fixed.

    Returns
    pandas.DataFrame
        The DataFrame with fixed structure. If the original DataFrame structure could not be fixed 
        (i.e., no headers were found), a placeholder DataFrame is returned. This placeholder DataFrame 
        contains the known judgements and "data_unreadable" as the grade for each of them.
    """

    headers = ["judgement", "grade"]
   
    known_judgements = ["impact_of_leaders", 
                        "help_and_protection", 
                        "in_care_and_care_leavers",
                        "in_care",
                        "care_leavers",
                        "overall_effectiveness"]
    
    # Create a placeholder DataFrame
    corrected_df = pd.DataFrame({
        "judgement": known_judgements,
        "grade": ["data_unreadable"] * len(known_judgements)  # Fill with default "data_unreadable"
    })
    
    for i, row in df.iterrows():
        df.columns = [col.lower().strip() for col in df.columns] # coerce consistent
        if set(headers).issubset(row.values):
            # If we found headers on the row, locate those columns containing headers
            # We do this as some tables arrive with non-headers at row[0]/messed up df structure
            headers_location = [column for column in df.columns if any(header.lower() in str(cell).lower().strip() for cell in df[column] for header in headers)]

            
            # Trim the DataFrame down to only columns now identiied as containing headers
            df = df[headers_location]

            # Remove non-headers/previous rows and set correct row as new headers
            df = df.iloc[i+1:].reset_index(drop=True)
            df.columns = headers
            
            return df

    # If no known headers were found at all, return placeholder DataFrame
    return corrected_df



def fix_misalligned_judgement_table(df):
    """
    Fix misaligned judgement tables by removing unnecessary rows(e.g. multi-line judgements and grade) and assigning 
    grades to correct judgements.

    This function is designed to handle formatting issues encountered when extracting the inspection 
    tables from the report pdf files. Assigns grades to known judgements, ensuring the data is correctly aligned. 
    This is a revised approach from the previous 'search for' method applied as a result of the unreliable and mis-
    alligned structures encountered from the pdf extract. Unfortunately it's now more table structure dependent, 
    but has proven more reliable at getting more of the grades out. From ~90% to ~99% 

    Args:
        df (pandas.DataFrame): The DataFrame containing the judgement table. 
                               It must contain 'judgement' and 'grade' columns.

    Returns:
        pandas.DataFrame: A new DataFrame with the same columns as the input, 
                          but with grades assigned to known judgements and any misaligned rows removed. 

    Raises:
        ValueError: If the input DataFrame does not contain the necessary columns or the data cannot be processed correctly.
        
    Note:
        This function makes certain assumptions about the structure and contents of the input DataFrame, 
        including the presence of specific judgements and the order in which they appear.
        It also assumes that the 'overall effectiveness' judgement (if present) is the last relevant row in the DataFrame.
    """

    
    # shortened known judgements
    known_judgements = ["impact_of_leaders", 
                        "help_and_protection", 
                        # "in_care_and_care_leavers", # BAK PRe Jan 2023
                        "in_care",
                        "care_leavers",
                        "overall_effectiveness"]

    
    # Create a placeholder DataFrame
    corrected_df = pd.DataFrame({
        "judgement": known_judgements,
        "grade": [np.nan] * len(known_judgements)  # Fill with NaNs
    })

    # Grade cleaning
    # This is entirely over-kill/brute force, however some problematic nan string formats encountered. 
    # Lowercase all grades and remove any newline characters and leading/trailing white space
    df['grade'] = df['grade'].str.lower().str.replace('\n', '').str.strip()
    # Remove any non-alphanumeric and non-whitespace characters
    df['grade'] = df['grade'].str.replace(r'[^\w\s]', '')
    # Convert any empty strings to NaN
    df.loc[df['grade'] == '', 'grade'] = np.nan
    # Replace 'nan' strings with actual NaN values
    df['grade'].replace('nan', np.nan, inplace=True)
    

    # Note: This line might not be required anymore as potentially duplicated elsewhere. To review/remove.
    # + Some RI text has further comment that we don't want, i.e. 'RI, *to become good*'
    df['grade'] = df['grade'].replace("(?i).*requires improvement.*", "requires improvement", regex=True)
    ## end

    # Get grades IN ORDER (ignoring NaN values and 'nan' strings)
    # print(df['grade'].apply(repr).unique()) # TESTING - check for non-printing chars
    grades = [grade for grade in df['grade'].dropna().tolist() if grade != 'nan']

    # Find 'overall effectiveness' judgement and grade and remove that+subsequent row(s) from the original df
    # reduces chances of mis-alligning sub-grades (which often arrived from pdf with difficult formatting)
    oe_index = df[df['judgement'].str.contains('overall effectiveness', case=False, na=False)].index
    if not oe_index.empty:
        corrected_df.loc[corrected_df['judgement'] == 'overall_effectiveness', 'grade'] = df.loc[oe_index[0], 'grade']

        indices_to_remove = df.index[df.index > oe_index[0]]
        df = df.drop(indices_to_remove)


    # Assign grades to the correct judgements in placeholder df

    # New 2023 sub-judgements (split in-care / care leavers)
    corrected_df.loc[corrected_df['judgement'] == known_judgements[0], 'grade'] = grades[0] # "impact_of_leaders"
    corrected_df.loc[corrected_df['judgement'] == known_judgements[1], 'grade'] = grades[1] # "help_and_protection"
    corrected_df.loc[corrected_df['judgement'] == known_judgements[2], 'grade'] = grades[2] # "in_care" && "in_care_and_care_leavers"
    
    if len(grades) > 3:
        corrected_df.loc[corrected_df['judgement'] == known_judgements[3], 'grade'] = grades[3] # 'care_leavers"
    

    # # TESTING 180324 RH
    # print("START: test in fix_misalligned_judgement_table()")
    # if len(grades) > 4:
    #     print(f"New Post Jan2023 framework identified with {len(grades)} grades: {grades}")
    # elif len(grades) > 5:
    #     print(f"Unexpectedly high number of grades found ({len(grades)}). Investigate this: {grades}")
    # else:
    #     print(grades)

    return corrected_df




def extract_inspection_data_update(pdf_content):
    """
    Function to extract key details from inspection reports PDF.

    Args:
        pdf_content (bytes): The raw content of the PDF file to be processed. 

    Returns:
        dict: A dictionary containing the extracted details. The dictionary keys are as follows:
            - 'table_rows_found': Number of rows found in the table.
            - 'inspector_name': The name of the inspector.
            - 'overall_inspection_grade': The overall effectiveness grade.
            - 'inspection_start_date': The start date of the inspection.
            - 'inspection_end_date': The end date of the inspection.
            - 'inspection_framework': The inspection framework string.
            - 'impact_of_leaders_grade': The impact of leaders grade.
            - 'help_and_protection_grade': The help and protection grade.
            - 'in_care_grade': The in care grade.
            - 'care_leavers_grade': The care leavers grade.
            - 'sentiment_score': The sentiment score of the inspection report.
            - 'sentiment_summary': The sentiment summary of the inspection report.
            - 'main_inspection_topics': List of key inspection themes.
    
    Raises:
        ValueError: If the PDF content is not valid or cannot be processed correctly.
        
    Note:
        This function expects the input PDF to contain specific sections specifically
        the inspection judgements to be on page 1 (page[0]) 
        If the PDF structure is different, obv the function will need changing. 
    """

    # Create a file-like buffer for the PDF content
    with io.BytesIO(pdf_content) as buffer:
        # Read the PDF content for text extraction
        reader = PyPDF2.PdfReader(buffer)
        
        # Extract the first page of inspection report pdf
        # This to ensure when we iterate/search the summary table, chance of invalid table reduced
        first_page_text = reader.pages[0].extract_text()

        # Extract text from <all> pages in the pdf
        full_text = ''
        for page in reader.pages:
            full_text += page.extract_text()

        
        # # ################# #sentiment
        # # # dev-in-progress

        # # Generate inspection sentiment score
        # # 

        # #sentiment
        # # Call the get_sentiment_and_topics function
        # sentiment_val, key_inspection_themes_lst = get_sentiment_and_topics(buffer, report_sentiment_ignore_words)

        # # Convert val to a <general> sentiment text/str for (readable) reporting
        # sentiment_summary_str = get_sentiment_category(sentiment_val)

        # # #################
        # # # dev-in-progress
        
        # # # Call the updated get_sentiment** function # testing
        # # sentiment_val2, filtered_themes = get_sentiment_and_sentiment_by_theme(buffer, "leadership", "results", "management") # testing
        # # plot_filtered_topics(filtered_themes) # testing
        # # #################



        # Find the inspector's name using a regular expression
        match = re.search(r"Lead inspector:\s*(.+)", first_page_text)
        if match:
            inspector_name = match.group(1)
            
            inspector_name = inspector_name.split(',')[0].strip()       # Remove everything after the first comma (some contain '.., Her Majesty’s Inspector')
            inspector_name = inspector_name.replace("HMI", "").rstrip() # Remove "HMI" and any trailing spaces(some inspectors add this to name)

        else:
            inspector_name = None

        # Read the PDF and extract the table on the first page
        try:
            buffer.seek(0)  # Reset the buffer position to the beginning
            tables = tabula.read_pdf(buffer, pages=1, multiple_tables=True)
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
            tables = []



    # Find the inspection dates
    #

    date_match = re.search(r"Inspection dates:\s*(.+)", first_page_text) # use regular expression

    if date_match:
        # IF there was date data


        inspection_dates = date_match.group(1).strip()
            
        # Some initial clean up based on historic data obs
        inspection_dates = inspection_dates.replace(".", "")
        inspection_dates = inspection_dates.replace("\u00A0", " ") # Remove non-breaking space (Seen in nottingham report)
        inspection_dates = re.sub(r"[\u2012\u2013\u2014\u2212\-]+", " to ", inspection_dates) # replace en dash char ("\u2013"), em dash ("\u2014"), or ("-") 
        inspection_dates = inspection_dates.split("and")[0].strip() # Need this because we have such as :
                                                                    # "8 July 2019 to 12 July 2019 and 7 August 2019 to 8 August 2019"
                                                                    # E.g. Derbyshire
        inspection_dates = re.sub(r'(\d)\s(\d)', r'\1\2', inspection_dates) # Fix white spaces between date numbers e.g. "wiltshire,	1 9 June 2019"



        if isinstance(inspection_dates, str):
            # data was as expected
            year_match = re.search(r"\d{4}", inspection_dates)
            if year_match:
                year = year_match.group(0) # get single copy of yyyy

                # Now remove the year from the inspection_dates string
                inspection_dates_cleaned = inspection_dates.replace(year, "").strip()

            else:
                # We had inspection_dates data but no recognisable year
                year = None
                inspection_dates_cleaned = inspection_dates.strip()

        else:
            # spurious data
            # inspection_dates arrived with non-str, set default val
            print("Error: inspection_dates is not a string. Type is", type(inspection_dates))
            inspection_dates_cleaned = None 


        # Now that we have already removed/cleaned those with 'and .....'
        # Split the inspection_dates_cleaned string using ' to ' as the delimiter and limit the number of splits to 1
        date_parts = inspection_dates_cleaned.split(' to ', maxsplit=1) # Assumption - expect only 1 instance of 'to' between date vals
        

  
        # Get the seperate inspection date(s) 
        start_date = date_parts[0].strip()
        end_date = date_parts[1].strip() if len(date_parts) > 1 else None
        
        # Check if the month text is written in *both* the date strings
        # Required work-around as Ofsted reports contain inspection date strings in multiple formats (i/ii/iii...)
        #   i)      "15 to 26 November"  
        #   ii)     "28 February to 4 March" or "8 October to 19 October" (majority)
        #   iii)    ['8 July ', '12 July   and 7 August  to'] (*recently seen)
        #   iv)     "11 September 2017 to 5 October 2017" (double year)
        #   v)      "Inspection dates: 19 November–30 November 2018" (Bromley)
        if len(start_date) <= 2: # i.e. do we only have a date with no month text
            inspection_month = end_date.split()[1]
            start_date = f"{start_date} {inspection_month}"


        # Append the inspection year to the start_date and end_date
        # Note: This needs further work/decision on how to handle possible 'None' years where no recognisable was found
        start_date_str = f"{start_date} {year}"
        end_date_str = f"{end_date} {year}" if end_date else None


        # format current str dates (as dt objects)
        start_date_formatted = parse_date(start_date_str, '%d %B %Y') #  convert from '8 January 2021' (str)
        end_date_formatted = parse_date(end_date_str, '%d %B %Y')

        # calculate inspection duration and return framework string
        # Note: Problems arising here generally relate to the end_date extraction from pdf
        inspection_framework_str = get_framework_type(start_date_formatted, end_date_formatted, short_inspection_threshold, standard_inspection_threshold)

    else:
        # unable to extract the data or didnt exist
        start_date_formatted = None
        end_date_formatted = None
        inspection_framework_str = None


    # Extract inspection judgements/grades
    #

    # Can be multiple tables on page 1(dodgy pdf formatting), ensure we only look at the 1st. 
    # 
    df = pd.DataFrame(tables[0])

    # Some initial clean-up / consistency checks
    df.columns = [col.lower().strip() for col in df.columns] # coerce consistent (headers)
    df = df.astype(str).applymap(lambda s: s.lower()) # coerce consistent (data+types)
    df = df.replace('\r', ' ', regex=True)


    # Check/enforce the expected grades table structure exists
    #

    # Expected headers exist?
    if not set(["judgement", "grade"]).issubset(df.columns):
        # They dont't, so re-allign structure or replace(last resort
        df = fix_invalid_judgement_table_structure(df)
        #  If the df structure is unrecognisable/unfixable, a placeholder df with dummy vals is returned


    # We have a great deal of messy extracted data
    # incl multi-line judgement strings that don't line up with grade. Need to address this.
    df = fix_misalligned_judgement_table(df)  
 
    # Short-term fix
    # We have some remaining known anomolies remaining in grade value structure
    # This is a not-ideal brute force fix for those
    columns_to_replace_grade_val = ['grade', 'overall_effectiveness', 'impact_of_leaders', 'help_and_protection', 'in_care', 'care_leavers', 'in_care_and_care_leavers']

    for column in columns_to_replace_grade_val:
        # handle just in-case we have a column naming mis-match 
        if column in df.columns:
            df[column] = df[column].replace({r'\b(be good\w*)\b': 'requires improvement', '(?i)nan': 'data_unreadable'}, regex=True)
        else:
            # [TESTING]
            # print(f"Column '{column}' not found in the DataFrame.")
            # print(df.columns)

            # Log the column names instead of printing
            logging.warning(f"Inspection date {start_date_formatted} / Column '{column}' not found in the DataFrame.")
            logging.info(df.columns)


    # Get judgement-grades as dict
    inspection_grades_dict = dict(zip(df['judgement'], df['grade']))
    
    # Ensure not yet introduced judgement is consistent pre-introduction
    # new care_leavers judgement introduction date (1st January 2023)
    judgement_chg_date_care_leavers = parse_date("01 January 2023", '%d %B %Y')
    try:
        # start_date_formatted is valid and pre the judgement introduction date
        if start_date_formatted and start_date_formatted < judgement_chg_date_care_leavers:
            # replace with default str val if inspection pre-dates judgement type
            inspection_grades_dict['care_leavers'] = 'inspection_pre_dates_judgement' # reset/coerce consistency in val

    except TypeError: # invalid type
        print("Date comparison failed due to invalid input.")


    return {
        # main inspection details
        'inspector_name':           inspector_name, 
        'overall_inspection_grade': inspection_grades_dict['overall_effectiveness'],
        'inspection_start_date':    start_date_formatted,
        'inspection_end_date':      end_date_formatted,
        'inspection_framework':     inspection_framework_str,
        'impact_of_leaders_grade':  inspection_grades_dict['impact_of_leaders'],
        'help_and_protection_grade': inspection_grades_dict['help_and_protection'],
        'care_leavers_grade':       inspection_grades_dict['care_leavers'], 
        'in_care_grade':            inspection_grades_dict['in_care'],                              

        # #sentiment
        # # inspection sentiments (in progress)
        # 'sentiment_score':          round(sentiment_val, 4), 
        # 'sentiment_summary':        sentiment_summary_str,
        # 'main_inspection_topics':   key_inspection_themes_lst,

        'table_rows_found':len(df)
        }







def process_provider_links(provider_links):
    """
    Processes provider links and returns a list of dictionaries containing URN, local authority, and inspection link.

    Args:
        provider_links (list): A list of BeautifulSoup Tag objects representing provider links.

    Returns:
        list: A list of dictionaries containing URN, local authority, inspection link, and, if enabled, additional inspection data.
    """
    
    data = []
    global pdf_data_capture # Bool flag
    global root_export_folder
    global inspections_subfolder


    for link in provider_links:
        # Extract the URN and provider name from the web link shown
        urn = link['href'].rsplit('/', 1)[-1]
        la_name_str = clean_provider_name(link.text.strip())


        clean_provider_dir = os.path.join(root_export_folder, inspections_subfolder, urn + '_' + la_name_str)
        provider_dir = os.path.join('.', root_export_folder, inspections_subfolder, urn + '_' + la_name_str)

        # Create the provider directory if it doesn't exist
        if not os.path.exists(provider_dir):
            os.makedirs(provider_dir)

        # Get the child page content
        child_url = 'https://reports.ofsted.gov.uk' + link['href']
        child_soup = get_soup(child_url)

        # Find all publication links in the provider's child page
        pdf_links = child_soup.find_all('a', {'class': 'publication-link'})

        # Initialise a flag to indicate if an inspection link has been found
        # Important: This assumes that the provider's reports are returned/organised most recent FIRST
        found_inspection_link = False

        # Iterate through the publication links
        for pdf_link in pdf_links:

            # Check if the current/next href-link meets the selection criteria
            # This block obv relies on Ofsted continued use of nonvisual element descriptors
            # containing the type(s) of inspection text. We use  "children's services inspection"

            nonvisual_text = pdf_link.select_one('span.nonvisual').text.lower().strip()

            # For now at least, web page search terms hard-coded. 
            if 'children' in nonvisual_text and 'services' in nonvisual_text and 'inspection' in nonvisual_text:

                # Create the filename and download the PDF (this filetype needs to be hard-coded here)
                filename = nonvisual_text.replace(', pdf', '') + '.pdf'


                pdf_content = requests.get(pdf_link['href']).content
                with open(os.path.join(provider_dir, filename), 'wb') as f:
                    f.write(pdf_content)


               # Extract the local authority and inspection link, and add the data to the list
                if not found_inspection_link:

                    # Capture the data that will be exported about the most recent inspection only
                    local_authority = provider_dir.split('_', 1)[-1].replace('_', ' ').strip()
                    inspection_link = pdf_link['href']
                    
                    # Extract the report published date
                    report_published_date_str = filename.split('-')[-1].strip().split('.')[0] # published date appears after '-' 
            
                    # get/format date(s) (as dt objects)
                    report_published_date = format_date(report_published_date_str, '%d %B %Y', '%d/%m/%y')

                    # Now get the in-document data
                    if pdf_data_capture:
                        # Opt1 : ~x4 slower runtime
                        # Only here if we have set PDF text scrape flag to True
                        # Turn this off, speeds up script if we only need the inspection documents themselves to be retrieved

               
                        # Scrape inside the pdf inspection reports
                        # inspection_data_dict = extract_inspection_data(pdf_content)
                        inspection_data_dict = extract_inspection_data_update(pdf_content)
                    

                        # Dict extract here for readability of returned data/onward

                        # inspection basics
                        overall_effectiveness = inspection_data_dict['overall_inspection_grade']
                        inspector_name = inspection_data_dict['inspector_name']
                        inspection_start_date = inspection_data_dict['inspection_start_date']
                        inspection_end_date = inspection_data_dict['inspection_end_date']
                        inspection_framework = inspection_data_dict['inspection_framework']
                        # additional inspection grades if available
                        impact_of_leaders_grade = inspection_data_dict['impact_of_leaders_grade']
                        help_and_protection_grade = inspection_data_dict['help_and_protection_grade']
                        # care_and_care_leavers_grade = inspection_data_dict['care_and_care_leavers_grade']
                        # # updates to reflect post jan 2023 summary changes
                        in_care_grade = inspection_data_dict['in_care_grade']
                        care_leavers_grade = inspection_data_dict['care_leavers_grade']

                        # #sentiment
                        # # NLP extract 
                        # sentiment_score = inspection_data_dict['sentiment_score']
                        # sentiment_summary = inspection_data_dict['sentiment_summary']
                        # main_inspection_topics = inspection_data_dict['main_inspection_topics']



                        # format dates for output                       
                        inspection_start_date_formatted = format_date_for_report(inspection_start_date, "%d/%m/%Y")
                        inspection_end_date_formatted = format_date_for_report(inspection_end_date, "%d/%m/%Y")

                        # Format the provider directory as a file path link (in readiness for such as Excel)
                        provider_dir_link = f"{provider_dir}"

                        
                        provider_dir_link = provider_dir_link.replace('/', '\\') # fix for Windows systems
                        
                        # TESTING
                        # print(f"{la_name_str}, {overall_effectiveness},{impact_of_leaders_grade}, {help_and_protection_grade}, {in_care_grade}, {care_leavers_grade}, {inspection_start_date_formatted}")

                        print(f"{local_authority}") # Gives listing console output during run in the format 'data/inspection reports/urn name_of_la'

                        data.append({
                                        'urn': urn,
                                        'local_authority': la_name_str,
                                        'inspection_link': inspection_link,
                                        'overall_effectiveness_grade': overall_effectiveness,
                                        'inspection_framework': inspection_framework,
                                        'inspector_name': inspector_name,
                                        'inspection_start_date': inspection_start_date_formatted,
                                        'inspection_end_date': inspection_end_date_formatted,
                                        'publication_date': report_published_date,
                                        'local_link_to_all_inspections': provider_dir_link,
                                        'impact_of_leaders_grade': impact_of_leaders_grade,
                                        'help_and_protection_grade': help_and_protection_grade,
                                        
                                        # 'care_and_care_leavers_grade': care_and_care_leavers_grade,
                                        'in_care_grade': in_care_grade, # This now becomes the care_and_care_leavers_grade if a pre Jan 2023 inspection
                                        'care_leavers_grade': care_leavers_grade,

                                        # #sentiment
                                        # 'sentiment_score': sentiment_score,
                                        # 'sentiment_summary': sentiment_summary,
                                        # 'main_inspection_topics': main_inspection_topics

                                    })
                        
                    else:
                        # Opt2 : ~x4 faster runtime
                        # Only grab the data/docs we can get direct off the Ofsted page 
                        data.append({'urn': urn, 'local_authority': local_authority, 'inspection_link': inspection_link})

                    
                    found_inspection_link = True # Flag to ensure data reporting on only the most recent inspection

    # print(data) # TEST 180324 RH
    # import sys
    # class UrnNotFoundException(Exception):
    #     pass

    # def check_urn_and_stop(data, target_urn):
    #     for item in data:
    #         if item['urn'] == target_urn:
    #             print(f"URN {target_urn} found. Stopping process.")
    #             raise UrnNotFoundException(f"URN {target_urn} found.")
    #     print("Target URN not found. Continuing process.")

    # try:
    #     check_urn_and_stop(data, '80490')
    # except UrnNotFoundException as e:
    #     print(e)
    #     sys.exit(1)  # Exit the script/program

    return data


def handle_pagination(soup, url_stem):
    """
    In the current version, this *not in use* as we have instead manipulated the site-php-search-url

    But if that method breaks in the future, we'll need to most likely return to a more usual scrape method with this. 
    Handles pagination for a BeautifulSoup object representing a web page with paginated content.
    
    Args:
        soup (bs4.BeautifulSoup): The BeautifulSoup object representing the web page.
        url_stem (str): The base URL to which the relative path of the next page will be appended.
        
    Returns:
        str: The full URL of the next page if it exists, otherwise None.
    """
    
    # Find the pagination element in the soup object
    pagination = soup.find('ul', {'class': 'pagination'})

    # Check if the pagination element exists
    if pagination:
        # Find the next page button in the pagination element
        next_page_button = pagination.find('li', {'class': 'next'})

        # Check if the next page button exists
        if next_page_button:
            # Extract the relative URL of the next page
            next_page_url = next_page_button.find('a')['href']
            
            # Return the full URL of the next page by appending the relative URL to the base URL
            return url_stem + next_page_url

    # Return None if there is no next page button or pagination element
    return None



def save_data_update(data, filename, file_type='csv', hyperlink_column = None):
    """
    Exports data to a specified file type.

    Args:
        data (DataFrame): The data to be exported.
        filename (str): The desired name of the output file.
        file_type (str, optional): The desired file type. Defaults to 'csv'.
        hyperlink_column (str, optional): The column containing folder names for hyperlinks. Defaults to None.

    Returns:
        None
    """
    if file_type == 'csv':
        filename_with_extension = filename + '.csv'
        data.to_csv(filename_with_extension, index=False)

    elif file_type == 'excel':
        filename_with_extension = filename + '.xlsx'

        # Create a new workbook and add a worksheet
        workbook = xlsxwriter.Workbook(filename_with_extension)
        sheet = workbook.add_worksheet('ofsted_cs_inspections_overview')  # pass the desired sheet name here

        hyperlink_col_index = data.columns.get_loc(hyperlink_column) if hyperlink_column else None

        # Define hyperlink format
        hyperlink_format = workbook.add_format({'font_color': 'blue', 'underline': 1})

        # Write DataFrame to the worksheet
        for row_num, (index, row) in enumerate(data.iterrows(), start=1):
            for col_num, (column, cell_value) in enumerate(row.items()):
                if hyperlink_col_index is not None and col_num == hyperlink_col_index:
                    # Add hyperlink using the HYPERLINK formula
                    link = f".\\{cell_value}"
                    sheet.write_formula(row_num, col_num, f'=HYPERLINK("{link}", "{cell_value}")', hyperlink_format)
                else:
                    sheet.write(row_num, col_num, str(cell_value))

        # Write header
        header_format = workbook.add_format({'bold': True})
        for col_num, column in enumerate(data.columns):
            sheet.write(0, col_num, column, header_format)

        # Save the workbook
        workbook.close()
    else:
        print(f"Error: unsupported file type '{file_type}'. Please choose 'csv' or 'excel'.")
        return

    print(f"{filename_with_extension} successfully created!")



def import_csv_from_folder(folder_name):
    """
    Imports a single CSV file from a local folder relative to the root of the script.

    The CSV file must be located in the specified folder. If multiple CSV files are found,
    a ValueError is raised. If no CSV files are found, a ValueError is raised.

    Parameters:
    folder_name (str): The name of the folder containing the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
    file_names = [f for f in os.listdir(folder_name) if f.endswith('.csv')]
    if len(file_names) == 0:
        raise ValueError('No CSV file found in the specified folder')
    elif len(file_names) > 1:
        raise ValueError('More than one CSV file found in the specified folder')
    else:
        file_path = os.path.join(folder_name, file_names[0])
        df = pd.read_csv(file_path)
        return df
    
    

def merge_and_select_columns(merge_to_df, merge_from_df, key_column, columns_to_add):
    """
    Merges two dataframes and returns a merged dataframe with additional columns from
    the second dataframe, without any duplicate columns. 

    Parameters:
    df1 (pandas.DataFrame): The first dataframe to merge.
    df2 (pandas.DataFrame): The second dataframe to merge.
    key_column (str): The name of the key column to merge on.
    columns_to_add (list): A list of column names from df2 to add to df1.

    Returns:
    pandas.DataFrame: A new dataframe with merged data from df1 and selected columns from df2.
    """
    merged = merge_to_df.merge(merge_from_df[columns_to_add + [key_column]], on=key_column)
    return merged



def reposition_columns(df, key_col, cols_to_move):
    """
    Move one or more columns in a DataFrame to be immediately to the right 
    of a given key column. 

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        key_col (str): The column that should be to the left of the moved columns.
        cols_to_move (list of str): The columns to move.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    # Check if the columns exist in the DataFrame
    for col in [key_col] + cols_to_move:
        if col not in df.columns:
            raise ValueError(f"{col} must exist in the DataFrame.")

    # Get a list of the column names
    cols = df.columns.tolist()

    # Find the position of the key column
    key_index = cols.index(key_col)

    # For each column to move (in reverse order)
    for col_to_move in reversed(cols_to_move):
        # Find the current index of the column to move
        col_index = cols.index(col_to_move)

        # Remove the column to move from its current position
        cols.pop(col_index)

        # Insert the column to move at the position immediately after the key column
        cols.insert(key_index + 1, col_to_move)

    # Return the DataFrame with reordered columns
    return df[cols]


### Start of GEO data proc
#
def read_json_to_dataframe(file_path, exclude_fields=None):
    """
    Convert a JSON file containing geo data into a pandas DataFrame.
    
    This function specifically handles GeoJSON formatted files. Made to allow the
    import of LA boundaries/ONS codes that can be then combined with the inspection 
    report data towards applying into such as a Chloropleth/mapping visualisations. 

    Parameters:
    - file_path (str): Path to the JSON file to be read.
    - exclude_fields (list, optional): A list of keys to exclude from the 
      properties of the features. Default is None.
    
    Returns:
    - pd.DataFrame: A DataFrame with each row being the properties of a 
      feature and an additional column for coordinates. All column headers 
      will be in lowercase.

    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    features = data['features']
    rows = []
    for feature in features:
        properties = feature['properties']
        geometry = feature['geometry']
        coordinates = geometry['coordinates']

        if exclude_fields:
            properties = {key: value for key, value in properties.items() if key not in exclude_fields}

        rows.append({**properties, 'coordinates': coordinates})

    df = pd.DataFrame(rows)

    # Force column headers to lowercase
    df.columns = df.columns.str.lower()

    return df


def replace_empty_ladcode_values(df, col1, col2):
    """
    Replace empty or NaN values in a specified DataFrame column with values from another column.
    We need to replace null LADCODE/LAD23CD from any geospacial data added as those rows will later fail if 
    not addressed. Some LA's have boundary data that does not conform to a single LAD23CD, so we 
    replace it with a value from another col, here that usually equates to ONS rgn22cd or ltla23cd

    Parameters:
    - df (pd.DataFrame): The DataFrame in which the replacement operation will be performed.
    - col1 (str): The name of the column to check for empty or NaN values.
    - col2 (str): The name of the column from which replacement values will be taken.

    Returns:
    - pd.DataFrame: The modified DataFrame with replaced values in `col1`.
    """
    df[col1] = df.apply(lambda row: row[col2] if pd.isnull(row[col1]) or row[col1] == '' else row[col1], axis=1)
    return df
### End of GEO data proc
#

def save_to_html(data, column_order, local_link_column=None, web_link_column=None):
    """
    Exports data to an HTML table.

    Args:
        data (DataFrame): The data to be exported.
        column_order (list): List of columns in the desired order.
        hyperlink_column (str, optional): The column containing hyperlinks. Defaults to None.

    Returns:
        None
    """
    # Define the page title and introduction text
    page_title = "Ofsted ILACS Summary"
    intro_text = (
        'Summarised outcomes of published short and standard ILACS inspection reports by Ofsted, refreshed daily.<br/>'
        'An expanded version of the shown summary sheet, refreshed concurrently, is available to '
        '<a href="ofsted_childrens_services_overview.xlsx">download here</a> as an .xlsx file. '
        '<br/>Data summary is based on the original <i>ILACS Outcomes Summary</i> published periodically by the ADCS: '
        '<a href="https://adcs.org.uk/inspection/article/ilacs-outcomes-summary">https://adcs.org.uk/inspection/article/ilacs-outcomes-summary</a>. '
        '<a href="https://github.com/data-to-insight/ofsted-ilacs-scrape-tool/blob/main/README.md">Read the tool/project background details and future work.</a>.'
    )

    disclaimer_text = (
        'Disclaimer: This summary is built from scraped data direct from https://reports.ofsted.gov.uk/ published PDF inspection report files. '
        'As a result of the nuances|variance within the inspection report content or pdf encoding, we\'re noting some problematic data extraction for a small number of LAs*.<br/> '
        '*LA extraction issues: southend-on-sea, [overall, help_and_protection_grade,care_leavers_grade], nottingham,[inspection_framework, inspection_date], redcar and cleveland,[inspection_framework, inspection_date], knowsley,[inspector_name], stoke-on-trent,[inspector_name]<br/>'
        '<a href="mailto:datatoinsight.enquiries@gmail.com?subject=Ofsted-Scrape-Tool">Feedback</a> on specific problems|inaccuracies|suggestions welcomed.*'
    )

    data = data[column_order]

    # Convert specified columns to title case
    title_case_cols = ['local_authority', 'inspector_name']
    for col in title_case_cols:
        if col in data.columns:
            data[col] = data[col].str.title()

    # Temporary removal (#TESTING) for clarity | fixes
    cols_to_drop = ['local_link_to_all_inspections', 'inspectors_inspections_count']
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(columns=col)


    # # If a local link column is specified, convert that column's values to HTML hyperlinks
    # # Displaying only the filename as the hyperlink text
    # if local_link_column:
    #     data[local_link_column] = data[local_link_column].apply(lambda x: '<a href="' + x + '">all_reports\\' + x.split("\\")[-1] + '</a>')


    # If a web link column is specified, convert that column's values to HTML hyperlinks
    # Shortening the hyperlink text by taking the part after the last '/'
    if web_link_column:
        data[web_link_column] = data[web_link_column].apply(lambda x: f'<a href="{x}">ofsted.gov.uk/{x.rsplit("/", 1)[-1]}</a>')

    # Convert column names to title/upper case
    data.columns = [c.replace('_', ' ').title() for c in data.columns]
    data.rename(columns={'Ltla23Cd': 'LTLA23CD', 'Urn': 'URN'}, inplace=True)


    # Generate 'Most-recent-reports' list (last updated list)
    # Remove this block if running locally (i.e. not in GitCodespace)
    # 
    # Obtain list of those inspection reports that have updates
    # Provides easier visual on new/most-recent on refreshed web summary page

    # specific folder to monitor for changes
    inspection_reports_folder = 'export_data/inspection_reports'

    try:
        # Init the repo object (so we know starting point for monitoring changes)
        repo = git.Repo(repo_path) 
    except Exception as e:
        print(f"Error initialising defined repo path for inspection reports: {e}")
        raise
    
    try:
    # Get current status of repo
        changed_files = [item.a_path for item in repo.index.diff(None) if item.a_path.startswith(inspection_reports_folder)]
        untracked_files = [item for item in repo.untracked_files if item.startswith(inspection_reports_folder)]

        # Combine tracked and untracked changes
        all_changed_files = changed_files + untracked_files

        # Remove the inspection_reports_folder path prefix from the file paths
        las_with_new_inspection_list = [os.path.relpath(file, inspection_reports_folder) for file in all_changed_files]

        # Remove "/children's services inspection" and ".pdf" from each list item string
        # overwrite with cleaned list items. 
        las_with_new_inspection_list = [re.sub(r"/children's services inspection|\.pdf$", "", file) for file in las_with_new_inspection_list]

        # # Verification output only
        # print("Changed files:", changed_files)
        # print("Untracked files:", untracked_files)
        # print("All changed files:", all_changed_files)
        print("Last updated list:", las_with_new_inspection_list)

    except Exception as e:
        print(f"Error processing repository: {e}")
        raise

# end of most-recent-reports generate
# Note: IF running this script locally, not in Git|Codespaces - Need to chk + remove any onward use of var: las_with_new_inspection_list 

    

    # current time, add one hour to the current time to correct non-UK Git server time
    adjusted_timestamp_str = (datetime.now() + timedelta(hours=1)).strftime("%d %m %Y %H:%M")

    # init HTML content with title and CSS
    html_content = f"""
    <html>
    <head>
        <title>{page_title}</title>
        <style>
            .container {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 10pt;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 5px;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <h1>{page_title}</h1>
        <p>{intro_text}</p>
        <p>{disclaimer_text}</p>
        <p><b>Summary data last updated: {adjusted_timestamp_str}</b></p>
        <p><b>LA inspections last updated: {las_with_new_inspection_list}</b></p>
        <div class="container">
    """

    # Convert DataFrame to HTML table
    html_content += data.to_html(escape=False, index=False)

    # Close div and HTML tags
    html_content += "\n</div>\n</body>\n</html>"

    # Write to index.html
    with open("index.html", "w") as f:
        f.write(html_content)

    print("ILACS summary page as index.html successfully created.")






# #
# # #sentiment
# #  In development section re: sentiment and other analysis 
# #
# #

# # Sentiment analysis additional stop/ignore words
# # bespoke stop words list (minimise uneccessary common non-informative words in the sentiment analysis)

# report_sentiment_ignore_words = [
#     # Words related to the organisation and nature of the report
#     'ofsted', 'inspection', 'report', 

#     # Words related to the subjects of the report
#     'child', 'children', 'children\'s', 'young', 'people', 

#     # Words related to the services involved
#     'service', 'services', 'childrens services', 'social', 'care', 

#     # Words related to the providers of the services
#     'staff', 'workers', 'managers',

#     # Words related to performance and outcomes
#     'achievement', 'achievements', 'outcome', 'outcomes', 'performance', 
#     'improvement', 'improvements',

#     # Words related to measures and standards
#     'assessment', 'assessments', 'standard', 'standards', 
#     'requirement', 'requirements', 'grade', 'grades', 

#     # Words related to the local authority and policy
#     'local', 'authority', 'policy', 'policies', 

#     # Words related to specific aspects of care
#     'help', 'support', 'provision', 'safeguarding', 'families', 
#     'work', 'leavers',

#     # Other
#     'year'
# ]


def get_sentiment_and_topics(pdf_buffer, ignore_words=[]):
    """
    Analyse the sentiment and extract the top 3 topics from a PDF document.

    This function takes a file-like buffer containing a PDF document as input and
    performs the following tasks:
    1. Reads the content of the PDF file using the PyPDF2 library.
    2. Extracts the text from each page and concatenates it into a single string.
    3. Performs sentiment analysis on the extracted text using the TextBlob library.
       The sentiment polarity score ranges from -1 (most negative) to 1 (most positive).
    4. Identifies key themes or topics from the extracted text using the Latent Dirichlet
       Allocation (LDA) model from the Gensim library.
    5. Returns the sentiment polarity score and the top 3 topics extracted from the PDF file.

    Args:
        pdf_buffer (io.BytesIO): A file(-like) buffer containing the PDF content.
        ignore_words (list): A list of words to be ignored during sentiment analysis(so we can remove common words)

    Returns:
        tuple: A tuple containing the sentiment polarity score (float) and a list of
               the top 3 topics (strings).
    """

    # Read the PDF stuff
    reader = PyPDF2.PdfReader(pdf_buffer)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Perform sentiment analysis on the extracted text
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Identify key themes from the extracted text
    # First, preprocess the text by tokenising and removing stop words
    tokens = [word for sentence in blob.sentences for word in sentence.words]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update(ignore_words)  # Add the inspections bespoke ignore words to the set of stop words
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # N.B Might need a further preprocessing step to normalise punctuation variations in the above


    # Create a dictionary from the tokenised text
    dictionary = corpora.Dictionary([tokens])
    
    # Create a corpus from the dictionary and the tokenised text
    corpus = [dictionary.doc2bow(tokens)]
    
    # Create an LDA model from the corpus
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)
    
    # Get the top 3 topics from the LDA model
    topics = [lda_model.print_topic(topic_num) for topic_num in range(3)]

    return sentiment, topics




# This an updated/extended version of the above 
def get_sentiment_and_sentiment_by_theme(pdf_buffer, theme1, theme2, theme3):
    """
    ****In progress****

    Args:
        

    Returns:
        
    """

    # Read the PDF stuff
    reader = PyPDF2.PdfReader(pdf_buffer)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Perform sentiment analysis on the extracted text
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Identify key themes from the extracted text
    # First, preprocess the text by tokenising and removing stop words
    tokens = [word for sentence in blob.sentences for word in sentence.words]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Create a dictionary from the tokenised text
    dictionary = corpora.Dictionary([tokens])
    
    # Create a corpus from the dictionary and the tokenised text
    corpus = [dictionary.doc2bow(tokens)]


    # Create an LDA model from the corpus with a higher number of topics
    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary)
    
    # Get all topics from the LDA model
    all_topics = [lda_model.print_topic(topic_num) for topic_num in range(10)]

    # Define a function to calculate similarity between two strings
    def string_similarity(s1, s2):
        vectorizer = CountVectorizer().fit_transform([s1, s2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]

    # Filter topics based on the similarity to the provided theme strings
    filtered_topics = []
    themes = [theme1, theme2, theme3]
    for topic in all_topics:
        for theme in themes:
            if string_similarity(topic, theme) > 0.2:  # Adjust the threshold as needed
                filtered_topics.append(topic)
                break

    return sentiment, filtered_topics

def get_sentiment_category(sentiment):
    """
    Return the sentiment category based on the sentiment value.

    Args:
        sentiment (float): Sentiment value ranging from -1 (most negative) to 1 (most positive).

    Returns:
        str: The sentiment category.
    """

    if sentiment > 0.8:
        return "Sentiment very positive"
    elif 0.4 < sentiment <= 0.8:
        return "Sentiment positive"
    elif -0.4 <= sentiment <= 0.4:
        return "Sentiment neutral"
    elif -0.8 < sentiment <= -0.4:
        return "Sentiment negative"
    else:
        return "Sentiment very negative"


def extract_words(topic_string):
    # Quick fix for when the sentiment weights per topic word not wanted.
    words = re.findall(r'\*"(.*?)"', topic_string)
    return words


def plot_filtered_topics(filtered_topics):
    """
    Note: This only running if using func get_sentiment_and_sentiment_by_theme(pdf_buffer, theme1, theme2, theme3) 

    Visualise filtered inspection topics as a bar chart.

    This function takes a list of filtered topics as input and creates a bar chart
    to visualise the weighted words for each topic.

    Args:
        filtered_topics (list): List of filtered topics as strings.

    Returns:
        None
    """

    import matplotlib.pyplot as plt # (intrim impot placement)

    # extract words and their weights from a topic string
    def extract_words_weights(topic_string):
        words_weights = [ww.split('*') for ww in topic_string.split(' + ')]
        return [(float(weight.strip()), word.strip(" '\"")) for weight, word in words_weights]

    # Extract words and their weights from the filtered_topics
    topics_words_weights = [extract_words_weights(topic) for topic in filtered_topics]

    # Create the bar chart for each topic
    for idx, (words_weights, topic) in enumerate(zip(topics_words_weights, filtered_topics), 1):
        words, weights = zip(*words_weights)

        fig, ax = plt.subplots()
        ax.barh(words, weights)
        ax.set_xlabel('Weights')
        ax.set_title(f'Topic {idx}: {topic[:50]}...')
        ax.invert_yaxis()  # Invert y-axis to show higher weights at the top

        plt.show()


#
#
#  END : In development section re: sentiment and other analysis 
#
#






#
# Scrape Ofsted inspection report data
#
data = []
while True:
    # Fetch and parse the HTML content of the current URL
    soup = get_soup(url)
    
    # Find all 'provider' links on the page
    provider_links = soup.find_all('a', href=lambda href: href and '/provider/' in href)

    # Process the provider links and extend the data list with the results
    data.extend(process_provider_links(provider_links))

    
    # Since all results are on a single page, no need to handle pagination. 
    # Processing complete.   
    break



# Convert the 'data' list to a DataFrame
ilacs_inspection_summary_df = pd.DataFrame(data)


#
# Add in some additional simplistic calc metric(s) cols

# Median sentiment score for each inspector
ilacs_inspection_summary_df['inspector_name'] = (ilacs_inspection_summary_df['inspector_name']
                                                 .str.strip()
                                                 .str.lower()
                                                 .str.replace('  ', ' '))  # clean up double spaces

# #sentiment
# ilacs_inspection_summary_df['inspectors_median_sentiment_score'] = (ilacs_inspection_summary_df
#                                                          .groupby('inspector_name')['sentiment_score']
#                                                          .transform('median')
#                                                          .round(4))

ilacs_inspection_summary_df['inspectors_inspections_count'] = (ilacs_inspection_summary_df
                                                  .groupby('inspector_name')['inspector_name']
                                                  .transform('count'))

# #sentiment
# # re-organise column structure now with new col(s)
# key_col = 'sentiment_score'
# cols_to_move = ['inspectors_median_sentiment_score','inspectors_inspections_count']
# ilacs_inspection_summary_df = reposition_columns(ilacs_inspection_summary_df, key_col, cols_to_move)





# Data enrichment - import flat-file stored data 
#

# Enables broader potential onward usage/cross/backwards-compatible access 
# Note: Where possible, avoid any reliance on flat-file stored dynamic data! 
#       - This process idealy only for static data, or where obtaining specific data points in a dynamic manner isnt possble etc. 
#       - These just examples of potential enrichment use-cases




# Enrichment: LA codes
# Ofsted data centres on URN, but some might need historic 'LA Number'

# import the needed external/local data
local_authorities_lookup_df = import_csv_from_folder(import_la_data_path) # bring external data in


# Ensure key column consistency
key_col = 'urn'
ilacs_inspection_summary_df['urn'] = ilacs_inspection_summary_df['urn'].astype('int64')
local_authorities_lookup_df['urn'] = pd.to_numeric(local_authorities_lookup_df['urn'], errors='coerce')

# Define what data is required to be merged in
additional_data_cols = ['la_code', 'region_code', 'ltla23cd', 'stat_neighbours']
ilacs_inspection_summary_df = merge_and_select_columns(ilacs_inspection_summary_df, local_authorities_lookup_df, key_col, additional_data_cols)

# re-organise column structure now with new col(s)
ilacs_inspection_summary_df = reposition_columns(ilacs_inspection_summary_df, key_col, additional_data_cols)
## End enrichment 1 ##



# # Enrichment: Geospatial boundary data
# # Import and append geospatial boundaries data for each LA (geojson)

# # Removed until decision reached on handling duplicate LAD23 codes in data


# # import the needed external/local data
# exclude_fields = ['globalid', 'shape_length', 'shape_area'] # we dont need these
# json_df = read_json_to_dataframe(import_geo_data_path+geo_boundaries_filename, exclude_fields)

# # Ensure key column consistency
# key_col = 'lad23cd'
# json_df = replace_empty_ladcode_values(json_df, 'lad23cd', 'utla23cd') # for cases where ladcd doesnt exist for non-lad LA areas.

# # Define what data is required to be merged in
# additional_data_cols = ['lad23cd', 'bng_e', 'bng_n', 'long', 'lat','coordinates']
# ilacs_inspection_summary_df = merge_and_select_columns(ilacs_inspection_summary_df, json_df, key_col, additional_data_cols)

# # re-organise column structure now with new col(s)
# # geo cols can remain in appended positions
# ## End enrichment 2 ##



# Enrichment: Anything else to enrich...?
## Re-use/add in a further duplicate of the above structure with the new/required data file
## End enrichment 3 ##


# json_df[json_df['lad23cd'].duplicated()]
# print(json_df.columns)
# print(ilacs_inspection_summary_df.columns)





#
# Fix(tmp) towards resultant export data types/excel cols type or format

# 020523 - Appears as though this is not having the desired effect once export file opened in Excel.
# Needs looking at again i.e. Urn still exporting as 'text' column

ilacs_inspection_summary_df['urn'] = pd.to_numeric(ilacs_inspection_summary_df['urn'], errors='coerce')
ilacs_inspection_summary_df['la_code'] = pd.to_numeric(ilacs_inspection_summary_df['la_code'], errors='coerce')
ilacs_inspection_summary_df['inspectors_inspections_count'] = pd.to_numeric(ilacs_inspection_summary_df['inspectors_inspections_count'], errors='coerce').fillna(0).astype(int)
# end tmp fix






# Export summary data (visible outputs)
#

# EXCEL Output
# Also define the active hyperlink col if exporting to Excel
save_data_update(ilacs_inspection_summary_df, export_summary_filename, file_type=export_file_type, hyperlink_column='local_link_to_all_inspections')


# WEB Output
# Set up which cols to take forward onto the web front-end(and order of)
# Remove for now until link fixed applied: 'local_link_to_all_inspections',
column_order = [
                'urn','la_code','region_code','ltla23cd','local_authority',
                'inspection_link','overall_effectiveness_grade','inspection_framework','inspector_name',
                'inspection_start_date',
                'local_link_to_all_inspections',
                'impact_of_leaders_grade','help_and_protection_grade','in_care_grade','care_leavers_grade',
                'inspectors_inspections_count'
                ]

save_to_html(ilacs_inspection_summary_df, column_order, local_link_column='local_link_to_all_inspections', web_link_column='inspection_link')


print("Last output date and time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


