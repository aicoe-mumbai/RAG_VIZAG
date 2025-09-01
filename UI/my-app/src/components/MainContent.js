
import React, { useState, useEffect, useRef, useCallback} from 'react';
import { useNavigate } from 'react-router-dom';
import './MainContent.css';
import PromptResponseCard from './main_sub_components/PromptResponseCard';
import ChatControls from './main_sub_components/ChatControls';
import NavBar from './NavBar';


function MainContent({ setHistory, selectedSessionId, resetSelectedSessionId, selectedFiles, userfile, mode}) {
  const [prompt, setPrompt] = useState('');
  const [responses, setResponses] = useState([]);
  const [editingIndex, setEditingIndex] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const apiUrl = process.env.REACT_APP_API_URL;
  const token = sessionStorage.getItem('authToken');
  const navigate = useNavigate();
  


  // Function to handle input change
  const handleInputChange = (e) => {
    setPrompt(e.target.value);
  };

  // Function to handle new chat
  const handleNewChat = () => {
    setResponses([]);
    setPrompt('');
    setEditingIndex(null);
    setSessionId(null);
    resetSelectedSessionId();
  };

  const handleSendPrompt = async (promptToSend) => {
    if (!promptToSend) return;
  
    const newResponseEntry = {
      prompt: promptToSend,
      response: '',
      loading: true,
    };
    
    // Add a new response entry with loading state to the responses array
    setResponses((prevResponses) => [...prevResponses, newResponseEntry]);
    setPrompt('');
    try {
      const fileNames = selectedFiles.map((file) => file);
      const currentPath = window.location.pathname;

      let mode = '';
      if (currentPath.includes('/qa/dashboard')) {
        mode = 'qa';
      } else if (currentPath.includes('/chat/dashboard')) {
        mode = 'chat';
      }
      const requestData = {
        prompt: promptToSend,
        session_id: sessionId || selectedSessionId,
        file_names: fileNames,
        jwt_token: token,
        useruploadfile : userfile ? userfile.name : "",
        mode : mode
      };
      console.log("hello",requestData)
      const res = await fetch(`${apiUrl}/api/cohere/generate/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestData),
      });
      
      const historyId = res.headers.get('X-History-ID');
      const sessionIdFromHeader = res.headers.get('X-Session-ID');
    
  
      setSessionId(sessionIdFromHeader || selectedSessionId);
  
      // Handle streaming the partial responses
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let responseChunks = []; 
  
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
  
  
        // Decode and append the response chunk to the array
        const chunk = decoder.decode(value, { stream: true });
  
        if (chunk.trim()) {
          responseChunks = [...responseChunks, chunk]; 
        }
  
        if (chunk.includes('      '))  {
          done = true; 
          continue;
        }
  
        setResponses((prevResponses) => {
          const updatedResponse = {
            prompt: promptToSend,
            response: responseChunks.join(''), 
            loading: true, 
            id: historyId || null,
          };
    
          return [...prevResponses.slice(0, -1), updatedResponse];
        });
      }
      
        fetch(`${apiUrl}/api/history/?mode=${mode}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
        .then((res) => res.json())
        .then((data) => setHistory(data || []))
        .catch((error) => console.error('Error fetching history:', error));
  
      // Log and update the final response
      setResponses((prevResponses) => {
        const finalResponse = {
          prompt: promptToSend,
          response: responseChunks.join(''), 
          loading: false, 
          id: historyId || null,
        };
    
        return [...prevResponses.slice(0, -1), finalResponse];
      });

      setPrompt('');
  
    } catch (error) {
      console.error('Error:', error);
      setResponses((prevResponses) => {
        const errorResponse = {
          prompt: promptToSend,
          response: 'An error occurred',
          loading: false,
        };
  
        console.log('Error responses state:', [...prevResponses.slice(0, -1), errorResponse]);
        return [...prevResponses.slice(0, -1), errorResponse];
      });
  
      setPrompt('');
      alert("Session has ended. Please log in again."); 
      sessionStorage.clear();
      navigate('/login');
    }
  };

  
  // Initial call to fetch history when component loads
  useEffect(() => {
    fetch(`${apiUrl}/api/history/?mode=${mode}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then((res) => {
      if (res.status === 401) {  
          alert("Session has ended. Please log in again.");
          sessionStorage.clear();  
          window.location.href = '/login';  
          navigate('/login');
          return null;
      }
          return res.json(); 
      })
      .then((data) => setHistory(data || []))
      .catch((error) => console.error('Error fetching history:', error));
  }, [setHistory, apiUrl, token, navigate]);


const fetchSessionHistory = useCallback((session_id) => {
  fetch(`${apiUrl}/api/history/${session_id}/?mode=${mode}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
    .then((res) => res.json())
    .then((data) => {
      const sessionData = (data || []).map((item) => ({
        prompt: item.prompt,
        response: item.response,
        loading: false,
        id: item.id,
      }));

      setResponses(sessionData);
      setSessionId(session_id);
    })
    .catch((error) =>
      console.error('Error fetching session history:', error)
    );
}, [apiUrl, token, setResponses, setSessionId]);

useEffect(() => {
  if (selectedSessionId) {
    fetchSessionHistory(selectedSessionId);
  }
}, [selectedSessionId, fetchSessionHistory]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendPrompt(prompt);
    }
  };

   const responseEndRef = useRef(null);
   const prevResponseCountRef = useRef(0); 
 
   useEffect(() => {
     if (responses.length > prevResponseCountRef.current) {
       if (responseEndRef.current) {
         responseEndRef.current.scrollIntoView({ behavior: 'smooth' });
       }
     }
     prevResponseCountRef.current = responses.length;
   }, [responses]); 
 
   

  const handleSubmitEditedPrompt = async (newPrompt, index) => {
    let updatedResponses = [...responses];
    updatedResponses[index] = {
      ...updatedResponses[index],
      prompt: newPrompt,
      response: '',
      loading: true,
    };
    setResponses(updatedResponses);
  
    try {
      const fileNames = selectedFiles.map((file) => file);
  
      const requestData = {
        prompt: newPrompt,
        session_id: sessionId,
        file_names: fileNames,
        jwt_token: token,
        useruploadfile : userfile ? userfile.name : "",
        mode : mode
      };


      setEditingIndex(null);
      const res = await fetch(`${apiUrl}/api/cohere/generate/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestData),
      });
  
      const sessionIdFromHeader = res.headers.get('X-Session-ID');
     
      setSessionId(sessionIdFromHeader || sessionId);
  
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let responseChunks = [];
  
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        const chunk = decoder.decode(value, { stream: true });
  
        if (chunk.trim()) {
          responseChunks = [...responseChunks, chunk];
        }
   
        if (chunk.includes('      ') || doneReading) {
          done = true;
          continue;
        }
  
        setResponses((prevResponses) => {
          const updatedResponse = {
            ...prevResponses[index],
            response: responseChunks.join(' '),
            loading: true,
          };
          const newResponses = [...prevResponses];
          newResponses[index] = updatedResponse;
          return newResponses;
        });
      }
  
      setResponses((prevResponses) => {
        const finalResponse = {
          ...prevResponses[index],
          response: responseChunks.join(' '),
          loading: false,
        };
        const newResponses = [...prevResponses];
        newResponses[index] = finalResponse;
        return newResponses;
      });
  
     
    } catch (error) {
      updatedResponses[index].response = 'An error occurred';
      updatedResponses[index].loading = false;
      setResponses(updatedResponses);
    }
  };
  
  return (
    <div className="main-container">
      <NavBar />
      <div className="logo-container">
        {responses.length > 0 ? (
          
          <div className="responses-list"  >
            {responses.map((item, index) => {
  
              return (
                <PromptResponseCard
                  key={item.id}
                  item={item}
                  index={index}
                  editingIndex={editingIndex}
                  setEditingIndex={setEditingIndex}
                  handleSubmitEditedPrompt={handleSubmitEditedPrompt}
                  onSendPrompt={handleSendPrompt} 
                  mode={mode}
                />
              ); 
            })}
            <div ref={responseEndRef} />
          </div>
        ) : (
          <h1 className="saarthi-logo"> </h1>
        )}
      </div>

      <ChatControls
        prompt={prompt}
        handleInputChange={handleInputChange}
        handleKeyPress={handleKeyPress}
        handleSendPrompt={handleSendPrompt}
        handleNewChat={handleNewChat}
      />

      <p className="disclaimer">
        Correctness of response depends on the probabilistic nature of the model. For more precise and accurate information, please refer to the actual document!
      </p>
    </div>
  );
}

export default MainContent;