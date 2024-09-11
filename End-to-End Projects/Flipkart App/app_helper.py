from langchain_google_genai import GoogleGenerativeAI
from langchain_google_community import BigQueryVectorStore
from langchain.utilities import SQLDatabase
from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain_experimental.sql import SQLDatabaseChain

from google.cloud import bigquery
from google.oauth2 import service_account

import os
from dotenv import load_dotenv
load_dotenv()


def llm_db_chain():

    llm = GoogleGenerativeAI(model="models/text-bison-001",
                             google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.1, max_output_tokens=200)

    # Authenticate with Google Cloud (replace with your service account key path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

    project_id = 'gen-ai-432009'
    dataset_name = 'flipkart'

    # Create a BigQuery client
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json')
    client = bigquery.Client(
        credentials=creds, project=project_id, location='asia-south1')

    # Construct the database URI
    db_uri = f"bigquery://{project_id}/{dataset_name}"

    # Create the SQLDatabase object
    db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)

    # embeddings = GooglePalmEmbeddings(client=None, google_api_key=os.getenv('GOOGLE_API_KEY'))
    embeddings = HuggingFaceEmbeddings(client=None)

    vector_store = BigQueryVectorStore(
        project_id='gen-ai-432009',
        dataset_name='flipkart',
        table_name='clothing_data_vectors',
        location='asia-south1',
        credentials=creds,
        embedding=embeddings
    )

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store,
        k=3
    )

    # my sql based instruction prompt
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        # These variables are used in the prefix and suffix
        input_variables=["input", "table_info", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(
        llm, db, verbose=True, prompt=few_shot_prompt)

    return chain

    # input_dict = {"query": "Which is the best selling sub category?"}
    # response = chain.invoke(input_dict)
    # print(response)
