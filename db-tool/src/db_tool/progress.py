"""UI de progreso reutilizable"""

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


def create_progress_ui(console):
    """Crea interfaz de progreso Rich configurada"""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
