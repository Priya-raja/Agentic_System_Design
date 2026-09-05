from rich.console import Console
from rich.prompt import Prompt
from educosys_claude.config import config
from dotenv import load_dotenv


from educosys_claude.observability.logger import get_logger

load_dotenv()
logger = get_logger(__name__)
console = Console()

def run() -> None:
    logger.info("Eduosys Claude is starting...")

    console.print(
        "\n[bold blue]Educosys Claude[/bold blue]"
        " — RAG-powered code assistant"
    )

    console.print("Type [bold] '/exit' [/bold] to quit\n")

    while True:
        user_input = Prompt.ask("[bold green]'/exit' to quit\n[/bold green]")
        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("/exit", "/quit"):
            logger.info("Shutting down Eduosys Claude...")
            console.print("[dim]Goodbye![/dim]")
            break
        elif user_input.startswith("/ask "):
            question = user_input.removeprefix("/ask ").strip()

            logger.info("Ask command received: %s", question)

            console.print(
                f"[dim]Searching for: {question}...[/dim]"
            )

            # TODO: call retriever + LLM

        else:
            logger.warning(
                "Unknown command received: %s",
                user_input,
            )

            console.print("[yellow]Unknown command. Try:[/yellow]")
            console.print(
                "  [bold]/ask <question>[/bold] — ask a question"
            )
            console.print(
                "  [bold]/exit[/bold] — quit"
            )

    

if __name__ == "__main__":
    run()    