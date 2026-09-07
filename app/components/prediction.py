import streamlit as st

def render_prediction(result, profile=None):
    st.success(f"Assigned customer segment: Cluster {result['cluster']}")

    st.subheader("Cluster distances")

    for index, distance in enumerate(result["distances"]):
        st.write(
            f"Cluster {index}: {distance:.3f}"
        )
        st.progress(
            min(1.0, 1.0 / (1.0 + float(distance)))
        )

    if profile is not None:
        row = profile[
            profile["Cluster"] == result["cluster"]
        ]

        if not row.empty:
            record = row.iloc[0]

            st.subheader("Segment profile")

            st.write(
                f"Customers: {int(record['customers'])}"
            )
            st.write(
                f"Average age: {record['avg_age']:.1f}"
            )
            st.write(
                f"Average income: ${record['avg_income']:.1f}k"
            )
            st.write(
                f"Average spending score: {record['avg_spending']:.1f}/100"
            )
