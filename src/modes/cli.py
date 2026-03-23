"""Mode CLI interactif pour le RAG."""

import sys
import logging
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel

from ..rag_engine import create_engine
from ..prompt_templates import PREDEFINED_QUESTIONS

# Configuration du logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

console = Console()


def print_menu():
    """Affiche le menu principal."""
    console.print(Panel.fit(
        "[bold cyan]RAG Sanofi - Rapport Annuel 2022[/bold cyan]\n"
        "Choisissez une option:",
        border_style="cyan"
    ))
    console.print("  [1] Poser une question libre")
    console.print("  [2] Répondre aux 6 questions prédéfinies")
    console.print("  [3] Afficher les questions prédéfinies")
    console.print("  [4] Recherche sémantique (sans LLM)")
    console.print("  [5] Vérifier l'état du système")
    console.print("  [6] Quitter")


def ask_free_question(engine):
    """Pose une question libre."""
    question = Prompt.ask("[bold]Votre question[/bold]")
    
    console.print("\n[cyan]Recherche en cours...[/cyan]")
    result = engine.ask(question)
    
    console.print(Panel(
        f"[bold green]Réponse:[/bold green]\n{result['answer']}",
        title=f"Sources: {result['num_sources']} documents",
        border_style="green"
    ))
    
    console.print("\n[dim]Sources utilisées:[/dim]")
    for i, src in enumerate(result["sources"], 1):
        console.print(f"  {i}. Page {src['page']} (score: {src['score']:.2f})")


def answer_all_questions(engine):
    """Répond à toutes les questions prédéfinies."""
    console.print("\n[cyan]Traitement des 6 questions...[/cyan]\n")
    
    results = engine.answer_all_predefined(top_k=5)
    
    table = Table(title="Résultats des 6 questions", show_lines=True)
    table.add_column("ID", style="cyan", width=3)
    table.add_column("Catégorie", style="magenta")
    table.add_column("Réponse", max_width=60)
    
    for result in results:
        answer_preview = result["answer"][:57] + "..." if len(result["answer"]) > 60 else result["answer"]
        table.add_row(
            str(result["question_id"]),
            result["category"],
            answer_preview
        )
    
    console.print(table)
    
    for result in results:
        console.print(Panel(
            f"[bold]Q{result['question_id']}:[/bold] {PREDEFINED_QUESTIONS[result['question_id']-1]['question']}\n\n"
            f"[bold green]Réponse:[/bold green]\n{result['answer']}",
            title=f"Catégorie: {result['category']}",
            border_style="blue"
        ))
        console.print()


def show_predefined_questions():
    """Affiche les questions prédéfinies."""
    table = Table(title="Questions prédéfinies", show_lines=True)
    table.add_column("ID", style="cyan", width=3)
    table.add_column("Question", style="white")
    table.add_column("Catégorie", style="magenta")
    
    for q in PREDEFINED_QUESTIONS:
        question_preview = q["question"][:50] + "..." if len(q["question"]) > 50 else q["question"]
        table.add_row(str(q["id"]), question_preview, q["category"])
    
    console.print(table)


def search_semantic(engine):
    """Recherche sémantique uniquement."""
    query = Prompt.ask("[bold]Recherche[/bold]")
    
    results = engine.search_only(query, top_k=5)
    
    console.print(f"\n[cyan]{len(results)} résultats trouvés:[/cyan]\n")
    
    for i, doc in enumerate(results, 1):
        console.print(Panel(
            f"[bold]Page {doc['page']}[/bold] (score: {doc['score']:.2f})\n\n{doc['text'][:300]}...",
            border_style="yellow"
        ))


def health_check(engine):
    """Vérifie l'état du système."""
    status = engine.health_check()
    
    table = Table(title="État du système", show_lines=True)
    table.add_column("Composant", style="cyan")
    table.add_column("Status", style="magenta")
    
    for component, is_ok in status["checks"].items():
        status_icon = "[green]✓[/green]" if is_ok else "[red]✗[/red]"
        table.add_row(component, status_icon)
    
    console.print(table)
    console.print(f"\nDocuments dans la DB: [bold]{status['documents_count']}[/bold]")
    console.print(f"Status global: [bold]{status['status']}[/bold]")


def run_cli():
    """Lance le mode CLI interactif."""
    console.print("\n[bold cyan]Initialisation du moteur RAG...[/bold cyan]\n")
    
    try:
        engine = create_engine()
        engine.initialize()
        console.print("[green]✓ Moteur RAG initialisé[/green]\n")
    except Exception as e:
        console.print(f"[red]Erreur lors de l'initialisation: {e}[/red]")
        return
    
    while True:
        print_menu()
        choice = Prompt.ask("\n[bold]Choix[/bold]", choices=["1", "2", "3", "4", "5", "6"])
        
        if choice == "1":
            ask_free_question(engine)
        elif choice == "2":
            answer_all_questions(engine)
        elif choice == "3":
            show_predefined_questions()
        elif choice == "4":
            search_semantic(engine)
        elif choice == "5":
            health_check(engine)
        elif choice == "6":
            console.print("[cyan]Au revoir![/cyan]")
            break


if __name__ == "__main__":
    run_cli()