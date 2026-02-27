"""Gradio UI for RAG Playground."""
import sys
sys.path.insert(0, ".")

import gradio as gr
from PIL import Image
from src.rag_pipeline import RAGPipeline
from src.causal_chain import extract_causal_chains, format_full_output

# Initialize RAG pipeline
print("Initializing RAG Playground...")
rag = RAGPipeline()

if not rag._is_ingested:
    print("No index found. Running ingestion...")
    rag.ingest(max_samples=300)


def query_rag(question: str, image: Image.Image | None = None) -> tuple[str, str]:
    """
    Process a query through the RAG pipeline.

    Returns:
        tuple of (answer, sources)
    """
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = rag.query(question, image=image)

        # Format sources
        sources_text = "### Retrieved Sources\n\n"
        for i, (doc, meta, dist) in enumerate(zip(
            result.retrieval.documents,
            result.retrieval.metadatas,
            result.retrieval.distances,
        )):
            relevance = f"{(1 - dist) * 100:.1f}%"
            uid = meta.get("uid", "unknown")
            section = meta.get("section", "unknown")
            sources_text += f"**[Source {i+1}]** Report {uid} ({section}) - Relevance: {relevance}\n"
            sources_text += f"> {doc[:200]}...\n\n"

        if result.retrieval.image_description:
            sources_text += f"\n### Image Analysis\n{result.retrieval.image_description}"

        return result.answer, sources_text

    except Exception as e:
        return f"Error: {str(e)}", ""


def analyze_causal_chains(article: str) -> tuple[str, str, str]:
    """
    Extract causal chains from an article.

    Returns:
        tuple of (main_output, mermaid_diagram, raw_json)
    """
    if not article.strip():
        return "Please paste an article.", "", ""

    try:
        result = extract_causal_chains(article)
        main_output, mermaid_code, raw_json = format_full_output(result)
        return main_output, mermaid_code, raw_json
    except Exception as e:
        return f"Error: {str(e)}", "", ""


CAUSAL_CHAIN_EXAMPLES = [
    [
        "Climate change is causing glaciers to melt at an unprecedented rate. "
        "As glaciers melt, sea levels rise, threatening coastal communities. "
        "Rising sea levels lead to increased flooding in low-lying areas, "
        "which displaces populations and strains urban infrastructure. "
        "The displacement of populations creates refugee crises, "
        "putting pressure on neighboring regions' resources and social services."
    ],
    [
        "The rapid adoption of remote work during the pandemic reduced demand for "
        "commercial office space. Decreased occupancy rates led to falling commercial "
        "real estate prices in major cities. As property values declined, cities saw "
        "reduced tax revenue from commercial properties. Lower tax revenue forced "
        "budget cuts in public services, affecting transportation and infrastructure "
        "maintenance. Poor infrastructure further discouraged businesses from "
        "returning to downtown areas, creating a negative feedback loop."
    ],
    [
        "Overuse of antibiotics in livestock farming has accelerated the evolution of "
        "antibiotic-resistant bacteria. These resistant strains can transfer to humans "
        "through the food supply chain. As common antibiotics become ineffective, "
        "treating routine infections becomes more difficult and expensive. This increases "
        "hospital stays and healthcare costs, and in severe cases leads to higher "
        "mortality rates from previously treatable conditions."
    ],
]


# Build Gradio interface
with gr.Blocks(
    title="RAG Playground",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # RAG Playground
        """
    )

    with gr.Tabs():
        # --- Tab 1: RAG Query ---
        with gr.TabItem("RAG Query"):
            gr.Markdown(
                """
                Ask questions over your document corpus. Optionally upload an image
                for multimodal analysis.

                *This system retrieves relevant documents and uses AI to generate
                evidence-based answers with source citations.*
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Medical Image (optional)",
                        type="pil",
                        height=300,
                    )
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What does cardiomegaly look like on a chest X-ray?",
                        lines=3,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["What are the most common findings in chest X-rays?"],
                            ["Describe the typical appearance of pneumonia on chest radiograph."],
                            ["What does cardiomegaly indicate and how is it identified?"],
                            ["What are the signs of pleural effusion on X-ray?"],
                            ["How can you differentiate between consolidation and atelectasis?"],
                        ],
                        inputs=[question_input],
                        label="Example Questions",
                    )

                with gr.Column(scale=1):
                    answer_output = gr.Markdown(label="AI Answer")
                    sources_output = gr.Markdown(label="Sources & Evidence")

            submit_btn.click(
                fn=query_rag,
                inputs=[question_input, image_input],
                outputs=[answer_output, sources_output],
            )
            question_input.submit(
                fn=query_rag,
                inputs=[question_input, image_input],
                outputs=[answer_output, sources_output],
            )

        # --- Tab 2: Causal Chain Extraction ---
        with gr.TabItem("Causal Chain Extraction"):
            gr.Markdown(
                """
                Paste an article or text below to extract cause-and-effect
                relationships and visualize causal chains.

                The system will identify causal pairs, link them into chains,
                and provide a visual diagram.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    article_input = gr.Textbox(
                        label="Article / Text",
                        placeholder="Paste your article here...",
                        lines=12,
                    )
                    extract_btn = gr.Button(
                        "Extract Causal Chains",
                        variant="primary",
                        size="lg",
                    )

                    gr.Examples(
                        examples=CAUSAL_CHAIN_EXAMPLES,
                        inputs=[article_input],
                        label="Example Articles",
                    )

                with gr.Column(scale=1):
                    causal_output = gr.Markdown(label="Causal Analysis")

            with gr.Row():
                with gr.Column():
                    mermaid_output = gr.Code(
                        label="Mermaid Diagram (copy to mermaid.live to visualize)",
                        language=None,
                    )
                with gr.Column():
                    json_output = gr.Code(
                        label="Raw JSON",
                        language="json",
                    )

            extract_btn.click(
                fn=analyze_causal_chains,
                inputs=[article_input],
                outputs=[causal_output, mermaid_output, json_output],
            )
            article_input.submit(
                fn=analyze_causal_chains,
                inputs=[article_input],
                outputs=[causal_output, mermaid_output, json_output],
            )

    gr.Markdown(
        """
        ---
        *Disclaimer: This tool is for educational and research purposes only.
        It should not be used for clinical diagnosis or treatment decisions.
        Always consult qualified medical professionals.*
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
