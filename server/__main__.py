from __future__ import annotations

import argparse
import sys

import uvicorn

from server.auth import hash_password
from server.config import Settings, load_settings
from server.db import bootstrap_database, session_scope
from server.models import User
from server.storage import exists as storage_exists
from server.workers.lease import fail_lost_jobs


def doctor(settings: Settings) -> int:
    bootstrap_database(settings)
    changed = fail_lost_jobs(settings)
    storage_exists("__doctor_probe__", settings)
    print(f"OK server (janitor={changed})")
    return 0


def reset_password(settings: Settings, email: str, password: str) -> int:
    with session_scope(settings) as db:
        user = db.query(User).filter_by(email=email.lower()).one_or_none()
        if user is None:
            print("Usuario nao encontrado")
            return 1
        user.password_hash = hash_password(password)
    print("Senha atualizada")
    return 0


def create_worker(settings: Settings, name: str) -> int:
    from server.models import WorkerNode, new_id

    with session_scope(settings) as db:
        worker = db.query(WorkerNode).filter_by(name=name).one_or_none()
        if worker is None:
            worker = WorkerNode(id=new_id(), name=name, status="offline", max_concurrent_jobs=1)
            db.add(worker)
        print(worker.id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="serve")
    parser.add_argument("subcommand", nargs="?")
    parser.add_argument("value", nargs="?")
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.lan:
        settings.bind = "lan"
    if args.workers != 1:
        print("Uvicorn deve rodar com exatamente 1 worker")
        return 2
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "admin" and args.subcommand == "reset-password" and args.value:
        password = input("Nova senha: ")
        return reset_password(settings, args.value, password)
    if args.command == "admin" and args.subcommand == "create-worker":
        return create_worker(settings, args.value or "admin-pc")
    uvicorn.run("server.app:app", host=settings.host, port=settings.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
