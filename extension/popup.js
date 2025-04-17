document.getElementById("saveBtn").addEventListener("click", async () => {
    const className = document.getElementById("className").value.trim();
  
    if (!className) {
      alert("Please enter a class name");
      return;
    }
  
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      args: [className],
      func: (className) => {
        const elements = document.getElementsByClassName(className);
        if (!elements.length) return null;
  
        let htmlContent = '';
        for (let el of elements) {
          htmlContent += el.outerHTML + '\n';
        }
        return htmlContent;
      }
    }, (results) => {
      if (!results || !results[0].result) {
        alert("Class not found on this page.");
        return;
      }
  
      const blob = new Blob([results[0].result], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      chrome.downloads.download({
        url: url,
        filename: "saved_content.html",
        saveAs: true
      });
    });
  });