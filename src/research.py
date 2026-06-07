import xml.etree.ElementTree as ET
import requests
import urllib.parse
from src.state import MedicalBoardState


def run_research_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 3: RESEARCH LIBRARIAN (DUAL-LAYER RAG) ---")

    # Extract unique final disease guesses from the doctor conversation records
    unique_diseases = set()
    for record in state.get("current_guesses", []):
        for candidate in record.get("candidates", []):
            unique_diseases.add(candidate.strip())

    print(f"Librarian analyzing following suggestions: {list(unique_diseases)}")

    evidence_index = {}

    # --- LAYER A: LOCAL ENCYCLOPEDIA (Orphanet XML Parsing) ---
    try:
        tree = ET.parse("data/en_product4.xml")
        root = tree.getroot()
        print("Successfully read local Orphadata XML index.")
    except Exception as e:
        print(f"Could not load local XML file: {e}. Defaulting to empty index storage.")
        root = None

    for disease in unique_diseases:
        evidence_index[disease] = {"orphadata_match": False, "pubmed_citations": []}

        # Look for the disease inside the XML data structures
        if root is not None:
            for disorder in root.findall(".//Disorder"):
                name_element = disorder.find("Name")
                if name_element is not None and disease.lower() in name_element.text.lower():
                    evidence_index[disease]["orphadata_match"] = True
                    print(f"[Orphadata Match Found]: {disease}")
                    break

        # --- LAYER B: LIVE DISCOVERY PORTAL (PubMed NCBI API) ---
        # Select up to 2 key symptom labels to build a strong Boolean string query
        keywords = list(state["hpo_labels"].values())[:2]
        query_string = f'"{disease}" AND (' + " OR ".join([f'"{kw}"' for kw in keywords]) + ')'
        encoded_query = urllib.parse.quote(query_string)

        try:
            # Step 1: Query Search endpoint to fetch matching Article IDs
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_query}&retmode=json&retmax=2"
            search_res = requests.get(search_url, timeout=10).json()
            id_list = search_res.get("esearchresult", {}).get("idlist", [])

            if id_list:
                # Step 2: Fetch details for those Article IDs
                fetch_ids = ",".join(id_list)
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={fetch_ids}&retmode=xml"
                fetch_res = requests.get(fetch_url, timeout=10).text

                # Parse out the titles as verifiable text citation arrays
                fetch_root = ET.fromstring(fetch_res)
                for article in fetch_root.findall(".//PubmedArticle"):
                    title = article.find(".//ArticleTitle")
                    pmid = article.find(".//PMID")
                    if title is not None and pmid is not None:
                        citation_token = f"PubMed ID {pmid.text}: {title.text}"
                        evidence_index[disease]["pubmed_citations"].append(citation_token)
                print(f"[PubMed Found {len(id_list)} links for]: {disease}")
        except Exception as err:
            print(f"Network error querying NCBI E-Utilities for {disease}: {err}")

    return {"evidence_payload": evidence_index}