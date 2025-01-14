import os

from ethikchat_argtoolkit.ArgumentGraph.update_templates_from_docs import update_templates_from_docs



DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data", "external", "argument_graphs")
print(DATA_DIR)


if __name__ == "__main__":
    update_templates_from_docs(scenario_dirs_path=DATA_DIR, google_docs_json_path=os.path.join("./", "google_docs.json"))
    # argument_graph_s1 = ResponseTemplateCollection.from_csv_files(os.path.join(DATA_DIR, "szenario_s1"))
