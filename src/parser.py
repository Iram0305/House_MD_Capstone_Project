import json
import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from src.state import MedicalBoardState


# A Pydantic schema tells the AI exactly how to format its response
class ExtractedSymptoms(BaseModel):
    symptoms: list[str] = Field(description="List of raw symptom phrases found in the text")


def run_parser_node(state: MedicalBoardState) -> dict:
    print("\n--- PHASE 1: PARSER AGENT ---")

    # 1. Initialize the LLM and force it to follow our strict schema
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(ExtractedSymptoms)

    # 2. Ask the LLM to extract raw text symptoms
    prompt = f"Extract all physical symptoms or clinical abnormalities from this patient story:\n\n{state['raw_narrative']}"
    result = structured_llm.invoke(prompt)
    raw_extracted = result.symptoms
    print(f"LLM extracted raw phrases: {raw_extracted}")

    # 3. Connect to our local Chroma database to match them to official HPO codes
    chroma_client = chromadb.PersistentClient(path="database/chroma_db")

    # Check if we already built the symptom index. If not, build it from hp-base.json
    try:
        collection = chroma_client.get_collection(name="hpo_symptoms")
    except Exception:
        print("Building local symptom index from hp-base.json. This happens only once...")
        collection = chroma_client.create_collection(name="hpo_symptoms")

        # Load your collected hp-base.json file
        with open("data/hp-base.json", "r") as f:
            hpo_data = json.load(f)

        ids = []
        documents = []
        # Look through the ontology structures
        for term in hpo_data.get("graphs", [{}])[0].get("nodes", []):
            if "id" in term and "lbl" in term and "HP_" in term["id"]:
                # Convert URL id to standard code format, e.g., HP:0001337
                hpo_id = term["id"].split("/")[-1].replace("_", ":")
                ids.append(hpo_id)
                documents.append(term["lbl"])

        # Batch insert into Chroma using standard embeddings
        emb = OpenAIEmbeddings()
        for i in range(0, len(ids), 500):
            collection.add(
                ids=ids[i:i + 500],
                documents=documents[i:i + 500]
            )

    validated_codes = []
    labels_map = {}

    # 4. Hard-gating lookup: Check if similarity score passes our 0.82 rule
    for symptom in raw_extracted:
        search_res = collection.query(query_texts=[symptom], n_results=1)
        if search_res["ids"] and search_res["distances"]:
            best_id = search_res["ids"][0][0]
            best_label = search_res["documents"][0][0]
            distance = search_res["distances"][0][0]
            # Chroma outputs distance. A distance < 0.18 roughly matches a similarity > 0.82
            if distance < 0.18:
                if best_id not in validated_codes:
                    validated_codes.append(best_id)
                    labels_map[best_id] = best_label

    print(f"Validated official HPO Codes: {validated_codes}")

    # Return updates to save directly into our Medical Board Clipboard state
    return {
        "validated_hpo_codes": validated_codes,
        "hpo_labels": labels_map,
        "debate_turn_counter": 1
    }