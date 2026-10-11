"""Streamlit frontend for the Task-06 customer churn inference API."""

import httpx
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"
PLAN_TYPES = ["Basic", "Pro", "Standard"]

st.set_page_config(
    page_title="Customer Churn Review",
    page_icon="📊",
    layout="centered",
)

st.title("Customer Churn Review")
st.caption("Submit a customer record to the existing churn inference API.")
st.info(
    "Educational decision-support demonstration only. Predictions are uncertain and "
    "must not automatically determine customer actions."
)

with st.sidebar:
    st.header("API connection")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).strip().rstrip("/")
    st.caption("Start the FastAPI server before requesting a prediction.")

with st.form("prediction_form"):
    st.subheader("Customer features")
    left, right = st.columns(2)
    with left:
        tenure_months = st.number_input(
            "Tenure (months)", min_value=0, max_value=1200, value=12, step=1
        )
        support_tickets = st.number_input(
            "Support tickets", min_value=0, max_value=100000, value=1, step=1
        )
        monthly_spend_inr = st.number_input(
            "Monthly spend (INR)", min_value=0.0, max_value=1_000_000_000.0,
            value=999.0, step=50.0,
        )
    with right:
        last_login_days = st.number_input(
            "Days since last login", min_value=0, max_value=100000, value=7, step=1
        )
        plan_type = st.selectbox("Plan type", options=PLAN_TYPES)

    submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

if submitted:
    if not api_url:
        st.error("Enter the FastAPI server URL in the sidebar.")
    else:
        payload = {
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "monthly_spend_inr": monthly_spend_inr,
            "last_login_days": last_login_days,
            "plan_type": plan_type,
        }
        try:
            with st.spinner("Requesting prediction..."):
                response = httpx.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=10.0,
                )
            if response.is_success:
                result = response.json()
                predicted_class = result["predicted_class"]
                readable_class = "Churn predicted" if predicted_class == 1 else "No churn predicted"
                st.subheader("Prediction")
                st.success(readable_class)

                probabilities = result.get("class_probabilities")
                if probabilities:
                    st.subheader("Class probabilities")
                    for class_label, probability in probabilities.items():
                        class_name = "Churn" if class_label == "1" else "No churn"
                        st.write(f"{class_name}: {probability:.1%}")
                        st.progress(max(0.0, min(float(probability), 1.0)))
                else:
                    st.caption("This model did not return probability scores.")

                st.caption(
                    "Probabilities are model outputs, not guarantees. Review the result "
                    "alongside relevant context before taking action."
                )
            elif response.status_code == 422:
                st.error("The API rejected the submitted features. Check the values and try again.")
            elif response.status_code == 503:
                st.error("The prediction model is currently unavailable. Check the API setup and try again.")
            else:
                st.error(f"The prediction service returned an error (HTTP {response.status_code}).")
        except httpx.ConnectError:
            st.error(
                "Could not reach the FastAPI service. Start it first, then try again. "
                "The default address is http://127.0.0.1:8000."
            )
        except httpx.TimeoutException:
            st.error("The API took too long to respond. Please try again.")
        except (httpx.HTTPError, ValueError, KeyError):
            st.error("The API response could not be processed. Please check the service and try again.")
