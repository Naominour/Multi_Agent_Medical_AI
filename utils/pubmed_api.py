import requests
import xml.etree.ElementTree as ET

def fetch_pubmed_info(question, api_key, retmax=3):
    # Fetches PubMed articles for medical evidence retrieval.
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {"db": "pubmed", "term": question, "retmode": "json", "retmax": retmax, "api_key": api_key}
    esearch_response = requests.get(esearch_url, params=esearch_params)
    esearch_data = esearch_response.json()
    id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
    return id_list if id_list else ["No relevant PubMed articles found."]
