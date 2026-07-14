import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from edu_claude.config import settings
from edu_claude.observability import configure_logging, get_logger

app = typer.Typer(
    name="edu-claude",
    help="A production-grade educational coding agent.",
)

console = Console()


def show_startup_panel() -> None:
    """Display application startup information in a Rich panel."""

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Application", settings.app.name)
    table.add_row("Version", settings.app.version)
    table.add_row("Environment", settings.app.environment)
    table.add_row("", "")
    table.add_row("LLM Provider", settings.llm.provider)
    table.add_row("LLM Model", settings.llm.model)

    console.print(
        Panel(
            table,
            title="[bold blue]Edu Claude[/bold blue]",
            subtitle="Production-grade educational coding agent",
            border_style="blue",
            expand=False,
        )
    )


@app.callback(invoke_without_command=True)
def main() -> None:
    """Start Edu Claude."""

    configure_logging()
    logger = get_logger(__name__)

   
    logger.info(
            "application_started",
            app_name=settings.app.name,
            version=settings.app.version,
            environment=settings.app.environment,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
        )

    show_startup_panel()