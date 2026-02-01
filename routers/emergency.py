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

@router.post("/trigger")
async def trigger_emergency(request: EmergencyRequest):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TWILIO_TO_NUMBER")

    if not all([account_sid, auth_token, from_number, to_number]):
        raise HTTPException(status_code=500, detail="Twilio credentials not configured.")

    try:
        client = Client(account_sid, auth_token)
        
        # 1. Send SMS (Always for SOS)
        sms_body = f"🚨 {request.severity.upper()} SOS ALERT! 🚨\nUser has triggered an emergency alert via MedGamma."
        if request.location:
            sms_body += f"\nLast Known Location: {request.location}"
            
        message = client.messages.create(
            body=sms_body,
            from_=from_number,
            to=to_number
        )
        print(f"✅ SMS Sent: {message.sid}")

        # 2. Make Call (Only for Critical)
        if request.severity == "critical":
            call = client.calls.create(
                 twiml='<Response><Say voice="alice">Hello Srijan, Emergency Alert. The user has triggered an SOS button in the Med Gamma application. Please check your messages immediately.</Say></Response>',
                 to=to_number,
                 from_=from_number
            )
            print(f"✅ Call Initiated: {call.sid}")
        else:
             print(f"ℹ️ Call skipped for severity: {request.severity}")

        return {"status": "success", "message": "Emergency alerts sent successfully."}

    except Exception as e:
        print(f"🔥 Twilio Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
