from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from app.core.config import Settings


logger = logging.getLogger(__name__)


class EmailDeliveryError(RuntimeError):
    pass


class EmailSender:
    def send(self, *, to: str, subject: str, text: str, html: str) -> None:
        raise NotImplementedError


class ConsoleEmailSender(EmailSender):
    def send(self, *, to: str, subject: str, text: str, html: str) -> None:
        del html
        logger.warning("Email to %s | %s\n%s", to, subject, text)


class BrevoEmailSender(EmailSender):
    def __init__(self, settings: Settings):
        if not settings.brevo_api_key:
            raise RuntimeError(
                "BREVO_API_KEY is required when EMAIL_BACKEND=brevo."
            )
        self.settings = settings

    def send(self, *, to: str, subject: str, text: str, html: str) -> None:
        payload = {
            "sender": {
                "name": self.settings.email_from_name,
                "email": self.settings.email_from,
            },
            "to": [{"email": to}],
            "subject": subject,
            "textContent": text,
            "htmlContent": html,
        }
        try:
            response = httpx.post(
                self.settings.brevo_api_url,
                headers={
                    "api-key": self.settings.brevo_api_key,
                    "accept": "application/json",
                },
                json=payload,
                timeout=self.settings.email_request_timeout,
            )
        except httpx.HTTPError as error:
            raise EmailDeliveryError("Brevo request failed.") from error
        if response.status_code >= 400:
            logger.error(
                "Brevo rejected an email: status=%s",
                response.status_code,
            )
            raise EmailDeliveryError("Brevo rejected the email.")


def build_email_sender(settings: Settings) -> EmailSender:
    backend = settings.email_backend.strip().casefold()
    if backend == "console":
        return ConsoleEmailSender()
    if backend == "brevo":
        return BrevoEmailSender(settings)
    raise RuntimeError(f"Unsupported EMAIL_BACKEND: {settings.email_backend!r}")


@dataclass(frozen=True)
class EmailMessage:
    to: str
    subject: str
    text: str
    html: str


def _frontend_link(settings: Settings, path: str, token: str) -> str:
    base = settings.app_base_url.rstrip("/")
    return f"{base}{path}?token={quote(token, safe='')}"


def verification_email(
    settings: Settings, to: str, token: str
) -> EmailMessage:
    link = _frontend_link(settings, "/verify-email", token)
    text = (
        "Confirm your FilmFinder email address.\n\n"
        f"Open this link to activate your account:\n{link}\n\n"
        f"This link expires in {settings.email_verification_ttl_hours} hours."
    )
    html = (
        "<p>Confirm your FilmFinder email address.</p>"
        f'<p><a href="{link}">Activate my account</a></p>'
        f"<p>This link expires in "
        f"{settings.email_verification_ttl_hours} hours.</p>"
    )
    return EmailMessage(
        to=to,
        subject="Confirm your FilmFinder account",
        text=text,
        html=html,
    )


def password_reset_email(
    settings: Settings, to: str, token: str
) -> EmailMessage:
    link = _frontend_link(settings, "/reset-password", token)
    text = (
        "Reset your FilmFinder password.\n\n"
        f"Open this link to choose a new password:\n{link}\n\n"
        f"This link expires in {settings.password_reset_ttl_minutes} minutes."
    )
    html = (
        "<p>Reset your FilmFinder password.</p>"
        f'<p><a href="{link}">Choose a new password</a></p>'
        f"<p>This link expires in "
        f"{settings.password_reset_ttl_minutes} minutes.</p>"
    )
    return EmailMessage(
        to=to,
        subject="Reset your FilmFinder password",
        text=text,
        html=html,
    )


def send_message(sender: EmailSender, message: EmailMessage) -> None:
    sender.send(
        to=message.to,
        subject=message.subject,
        text=message.text,
        html=message.html,
    )
