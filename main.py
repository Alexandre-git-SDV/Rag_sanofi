"""Point d'entrée principal pour le CLI."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.modes.cli import run_cli
from src.modes.json_export import export_to_json
from src.modes.api import create_app


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="RAG Sanofi - Interrogez le rapport annuel 2022"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "json", "api"],
        default="cli",
        help="Mode d'exécution"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Fichier de sortie pour le mode JSON"
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Exporter toutes les questions (mode JSON)"
    )
    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="+",
        help="IDs des questions spécifiques (mode JSON)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de documents à récupérer"
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="0.0.0.0",
        help="Hôte pour le mode API"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port pour le mode API"
    )
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        run_cli()
    
    elif args.mode == "json":
        output_path = Path(args.output) if args.output else Path("results.json")
        
        result = export_to_json(
            output_path=output_path,
            all_questions=args.all_questions,
            question_ids=args.question_ids,
            top_k=args.top_k
        )
        
        print(f"\n✓ Export terminé: {output_path}")
        print(f"  {result['summary']['total_questions']} questions traitées")
    
    elif args.mode == "api":
        import uvicorn
        app = create_app()
        
        print(f"Démarrage de l'API sur http://{args.api_host}:{args.api_port}")
        uvicorn.run(app, host=args.api_host, port=args.api_port)


if __name__ == "__main__":
    main()