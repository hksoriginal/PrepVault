import streamlit as st


class HTMLElementCreator:
    def create_leetcode_link_button(self, label, url):
        button_style = """
        <style>
        .button {
            border: 2px solid #D3D3D3; 
            background-color: transparent; /* No fill color */
            font-weight: bold;
            color: #D3D3D3; /* Text color matching the border */
            padding: 5px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 1px 2px;
            cursor: pointer;
            border-radius: 10px;
            transition: background-color 0.3s, color 0.3s; /* Transition for hover effect */
            animation: fadeInButton 1s ease-in-out; /* Animation for fading in */
        }
        .button:hover {
            background-color: #D3D3D3; /* Green fill on hover */
            color: black; /* Change text color on hover */
            animation: bounce 0.3s ease-in-out; /* Animation when hovering */
        }
        
        /* Fade-in animation for buttons */
        @keyframes fadeInButton {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Bounce animation for hover */
        @keyframes bounce {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        </style>
        """

        leetcode_icon = "https://upload.wikimedia.org/wikipedia/commons/1/19/LeetCode_logo_black.png?20191202080835"

        st.markdown(
            button_style
            + f'<a href="{url}" target="_blank"><button class="button"><img src="{leetcode_icon}" class="youtube-icon">{label}</button></a>',
            unsafe_allow_html=True,
        )

    def create_link_button(self, label, url):
        button_style = """
        <style>
        .button {
            border: 2px solid #D3D3D3; 
            background-color: transparent; /* No fill color */
            font-weight: bold;
            color: #D3D3D3; /* Text color matching the border */
            padding: 5px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 1px 2px;
            cursor: pointer;
            border-radius: 10px;
            transition: background-color 0.3s, color 0.3s; /* Transition for hover effect */
            animation: fadeInButton 1s ease-in-out; /* Animation for fading in */
        }
        .button:hover {
            background-color: #D3D3D3; /* Fill color on hover */
            color: black; /* Change text color on hover */
            animation: bounce 0.3s ease-in-out; /* Animation when hovering */
        }

        /* Fade-in animation for buttons */
        @keyframes fadeInButton {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Bounce animation for hover */
        @keyframes bounce {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        </style>
        """

        st.markdown(
            button_style
            + f'<a href="{url}" target="_blank"><button class="button">{label}</button></a>',
            unsafe_allow_html=True,
        )

    def create_youtube_link_button(self, label, url):
        button_style = """
        <style>
        .button {
            border: 2px solid #D3D3D3; 
            background-color: transparent; 
            font-weight: bold;
            color: #D3D3D3; 
            padding: 5px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-flex; /* Use flex to align icon and text */
            align-items: center; /* Center align items vertically */
            font-size: 16px;
            margin: 1px 2px;
            cursor: pointer;
            border-radius: 10px; 
            transition: background-color 0.3s, color 0.3s; /* Transition for hover effect */
            animation: fadeInButton 1s ease-in-out; /* Animation for fading in */
        }
        .button:hover {
            background-color: #D3D3D3; /* Fill color on hover */
            color: black; /* Change text color on hover */
            animation: bounce 0.3s ease-in-out; /* Animation when hovering */
        }
        .youtube-icon {
            width: 20px; /* Icon size */
            height: auto;
            margin-right: 10px; /* Space between icon and text */
        }

        /* Fade-in animation for buttons */
        @keyframes fadeInButton {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Bounce animation for hover */
        @keyframes bounce {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        </style>
        """

        youtube_icon = "https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png"
        st.markdown(
            button_style
            + f'<a href="{url}" target="_blank"><button class="button"><img src="{youtube_icon}" class="youtube-icon">{label}</button></a>',
            unsafe_allow_html=True,
        )
