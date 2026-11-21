export interface AuthUser {
  id: number;
  email: string;
  email_verified: boolean;
  created_at: string;
}

export interface MessageResponse {
  message: string;
}

export interface SignupResponse extends MessageResponse {
  verification_required: boolean;
}

export interface CsrfResponse {
  csrf_token: string;
}
