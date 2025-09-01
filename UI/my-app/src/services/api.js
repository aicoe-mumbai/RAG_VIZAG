const apiUrl = process.env.REACT_APP_API_URL;

export const refreshToken = async () => {
    const refreshToken = sessionStorage.getItem("refreshToken");

    if (!refreshToken) {
        console.error("No refresh token found, redirecting to login.");
        sessionStorage.clear();
        window.location.href = "/login"; 
        return null;
    }

    try {
        const response = await fetch(`${apiUrl}/api/token/refresh/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh: refreshToken }),
        });

        if (!response.ok) {
            throw new Error("Failed to refresh token");
        }

        const data = await response.json();
        sessionStorage.setItem("authToken", data.access); 
        return data.access;
    } catch (error) {
        console.error("Error refreshing token:", error);
        sessionStorage.clear();
        window.location.href = "/login"; 
        return null;
    }
};

export const fetchWithAuth = async (url, options = {}) => {
    let token = sessionStorage.getItem("authToken");

    if (!options.headers) {
        options.headers = {};
    }
    options.headers.Authorization = `Bearer ${token}`;

    let response = await fetch(url, options);

    if (response.status === 401) {
        console.warn("Access token expired, attempting refresh...");

        const newToken = await refreshToken();

        if (!newToken) return response; 

        options.headers.Authorization = `Bearer ${newToken}`;
        response = await fetch(url, options); 
    }

    return response;
};
