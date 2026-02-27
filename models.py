from pydantic import BaseModel, Field

class ClienteInput(BaseModel):
    Age:              int   = Field(..., ge=18, le=100, example=35)
    Sex:              int   = Field(..., ge=0,  le=1,   example=1)
    Job:              int   = Field(..., ge=0,  le=3,   example=2)
    Housing:          int   = Field(..., ge=0,  le=2,   example=1)
    Saving_accounts:  int   = Field(..., ge=0,  le=4,   example=1)
    Checking_account: int   = Field(..., ge=0,  le=3,   example=1)
    Credit_amount:    float = Field(..., gt=0,          example=1500)
    Duration:         int   = Field(..., gt=0,          example=12)
    Purpose:          int   = Field(..., ge=0,  le=7,   example=4)

class PrediccionOutput(BaseModel):
    risk:             str
    probability_good: float
    probability_bad:  float
    recommendation:   str

class BatchPrediccionOutput(BaseModel):
    total: int
    predicciones: list[PrediccionOutput]
