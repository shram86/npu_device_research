async function updateStatus() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) throw new Error("Status request failed");

    const data = await response.json();

    document.getElementById("camera-status").textContent =
      data.camera_connected ? "🟢 Connected" : "🔴 Disconnected";

    document.getElementById("motion-status").textContent =
      data.motion_detected ? "🔴 Detected" : "⚪ Idle";

    document.getElementById("recording-status").textContent =
      data.recording ? "🔴 Active" : "⚪ Idle";

    document.getElementById("storage-status").textContent =
      `${data.storage_gib.toFixed(3)} / ${data.storage_limit_gib} GiB`;

    document.getElementById("event-count").textContent = data.events;
  } catch (error) {
    document.getElementById("camera-status").textContent = "Unavailable";
    console.error("Status update failed:", error);
  }
}

updateStatus();
setInterval(updateStatus, 2000);

async function updateGallery() {
  try {
    const response = await fetch("/api/events?offset=0&limit=12", {
      cache: "no-store"
    });
    if (!response.ok) throw new Error("Gallery request failed");

    const data = await response.json();
    const gallery = document.getElementById("recordings-gallery");

    // Remove the initial empty-gallery message when events arrive.
    if (data.events.length) {
      const emptyMessage = gallery.querySelector(".muted");
      if (emptyMessage) emptyMessage.remove();
    }

    // Insert new events at the top, preserving older loaded events.
    for (const event of [...data.events].reverse()) {
      if (!loadedEventIds.has(event.id)) {
        gallery.prepend(createEventCard(event));
        loadedEventIds.add(event.id);
      }
    }

    // Remove cards whose recordings were deleted by storage cleanup.
    for (const card of gallery.querySelectorAll(".event")) {
      const img = card.querySelector("img");
      if (!img) continue;

      const match = img.getAttribute("src").match(
        /\/events\/([^/]+)\/peak\.jpg/
      );
      if (!match) continue;

      const id = match[1];

      // Check whether the snapshot still exists.
      const check = await fetch(img.getAttribute("src"), {
        method: "HEAD",
        cache: "no-store"
      });

      if (check.status === 404) {
        card.remove();
        loadedEventIds.delete(id);
      }
    }

    showMoreButton.style.display =
      data.total > loadedEventIds.size ? "block" : "none";

  } catch (error) {
    console.error("Gallery update failed:", error);
  }
}

setInterval(updateGallery, 5000);

// Track loaded events by ID so new recordings don't cause duplicates.
const loadedEventIds = new Set();
const showMoreButton = document.getElementById("show-more");

function syncLoadedEvents() {
  document.querySelectorAll("#recordings-gallery .event img").forEach(img => {
    const match = img.getAttribute("src").match(/\/events\/([^/]+)\/peak\.jpg/);
    if (match) loadedEventIds.add(match[1]);
  });
}

function createEventCard(event) {
  const card = document.createElement("div");
  card.className = "event";

  const imageLink = document.createElement("a");
  imageLink.href = `/events/${encodeURIComponent(event.id)}/peak.jpg`;
  imageLink.target = "_blank";

  const image = document.createElement("img");
  image.src = imageLink.href;
  image.alt = "Motion event snapshot";
  image.loading = "lazy";
  imageLink.appendChild(image);
  card.appendChild(imageLink);

  const timestamp = document.createElement("p");
  timestamp.textContent = event.timestamp;
  card.appendChild(timestamp);

  if (event.has_video) {
    const videoLink = document.createElement("a");
    videoLink.href = `/events/${encodeURIComponent(event.id)}/motion.avi`;
    videoLink.textContent = "Download video";
    videoLink.setAttribute("download", "");
    card.appendChild(videoLink);
  }

  return card;
}

async function loadMoreEvents() {
  showMoreButton.disabled = true;
  showMoreButton.textContent = "Loading...";

  try {
    syncLoadedEvents();

    // Fetch all event metadata to avoid offset shifts when new events arrive.
    let offset = 0;
    let hasMore = true;
    const unseen = [];

    while (hasMore && unseen.length < 12) {
      const response = await fetch(
        `/api/events?offset=${offset}&limit=48`,
        {cache: "no-store"}
      );
      if (!response.ok) throw new Error("Event request failed");

      const data = await response.json();

      for (const event of data.events) {
        if (!loadedEventIds.has(event.id)) {
          unseen.push(event);
          if (unseen.length >= 12) break;
        }
      }

      offset += data.events.length;
      hasMore = data.has_more;
      if (!data.events.length) break;
    }

    const gallery = document.getElementById("recordings-gallery");

    for (const event of unseen) {
      gallery.appendChild(createEventCard(event));
      loadedEventIds.add(event.id);
    }

    showMoreButton.style.display = hasMore || unseen.length === 12
      ? "block" : "none";
  } catch (error) {
    console.error("Failed to load events:", error);
  } finally {
    showMoreButton.disabled = false;
    showMoreButton.textContent = "Show more";
  }
}

showMoreButton.addEventListener("click", loadMoreEvents);
syncLoadedEvents();

// Check whether there are older events to display.
fetch("/api/events?offset=0&limit=1")
  .then(response => response.json())
  .then(data => {
    showMoreButton.style.display =
      data.total > loadedEventIds.size ? "block" : "none";
  })
  .catch(console.error);

updateGallery();
