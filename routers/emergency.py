from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/emergency", tags=["Emergency"])

class EmergencyRequest(BaseModel):
    type: str = "sos"  # 'sos' or 'checkin'
    location: str | None = None
    severity: str = "critical" # 'medium' (SMS only) or 'critical' (Call+SMS)

import threading

def _send_twilio_alert(type: str, severity: str, location: str | None, account_sid, auth_token, from_number, to_number):
    try:
        client = Client(account_sid, auth_token)
        
        # 1. Send SMS (Always for SOS)
        sms_body = f"🚨 {severity.upper()} SOS ALERT! 🚨\nUser has triggered an emergency alert via MedGamma."
        if location:
            sms_body += f"\nLast Known Location: {location}"
            
        message = client.messages.create(
            body=sms_body,
            from_=from_number,
            to=to_number
        )
        print(f"✅ SMS Sent: {message.sid}")

        # 2. Make Call (Only for Critical)
        if severity == "critical":
            call = client.calls.create(
                 twiml='<Response><Say voice="alice">Hello Srijan, Emergency Alert. The user has triggered an SOS button in the Med Gamma application. Please check your messages immediately.</Say></Response>',
                 to=to_number,
                 from_=from_number
            )
            print(f"✅ Call Initiated: {call.sid}")
    except Exception as e:
        print(f"🔥 Twilio Error: {e}")

@router.post("/trigger")
def execute_emergency_trigger(type: str, severity: str, location: str | None = None):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TWILIO_TO_NUMBER")

    if not all([account_sid, auth_token, from_number, to_number]):
        print("❌ Twilio credentials missing.")
        return "[SYSTEM]: Emergency credentials missing. Advise user to call emergency services manually."

    # Run in background thread so chat is not blocked
    thread = threading.Thread(
        target=_send_twilio_alert,
        args=(type, severity, location, account_sid, auth_token, from_number, to_number)
    )
    thread.daemon = True
    thread.start()
    
    # Return instruction for the LLM, NOT a message for the User
    return "[SYSTEM]: Emergency alert triggered in background. Focus on providing emotional support. Do NOT mention the alert trigger to the user."
@router.post("/trigger")
async def trigger_emergency(request: EmergencyRequest):
    result = execute_emergency_trigger(request.type, request.severity, request.location)
    if "Failed" in result:
         raise HTTPException(status_code=500, detail=result)
    return {"status": "success", "message": result}


