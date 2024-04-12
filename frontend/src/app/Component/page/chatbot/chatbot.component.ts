import { Component, OnInit } from '@angular/core';
import { chatroom } from '../../../model/chat';
import { UserService } from '../../../Services/user/user.service';
import { User } from '../../../model/user';
import { ChatService } from '../../../Services/chat/chat.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css'
})
export class ChatbotComponent implements OnInit {
  sendTerm: string = "";
  view = false;
  User!: User;
  chats?: chatroom[];
  profile = "../../../../assets/images/profile.png";
  constructor(private user: UserService, private charservices: ChatService, private router: Router) {
    this.User = user.currentUser;
    if (this.User.profile) {
      this.profile = this.User.profile
    } else {
      this.profile = "../../../../assets/images/profile.png";
    }
  }

  ngOnInit(): void {
    gsap.to(".chatlogo", {
      opacity: 1,
      right: "1rem",
      duration: 2
    })
    this.charservices.getchats(this.User.id).subscribe((chat) => {
      this.chats = chat
      console.log(chat)

    })
    window.onload = () => {
      var container = document.getElementById('container');
      if (container) {
        console.log(container.scrollTop)
        console.log(container.scrollHeight)

        container.scrollTop = container.scrollHeight;
        console.log(container.scrollTop)

      }
    }
  }

  send() {
    if (this.sendTerm != "") {
      const chat: chatroom = {
        id: "",
        name: this.User.name,
        send: this.sendTerm,
        responce: '',
        time: new Date(),
        userid: this.User.id
      }
      this.charservices.postchats(chat).subscribe((chat) => {
        this.chats = chat
        this.sendTerm = "";
        console.log(chat)
        this.mov()
      })
    }
  }
  mov() {
    setTimeout(() => {
      var container = document.getElementById('container');
      if (container) {
      container.scrollTop = container.scrollHeight;
      }
    }, 100);
}
  visiblechat() {
    this.user.userObeservable.subscribe((user) => {
      if (user.name) {
        if (this.view) {
          this.view = false;
        } else {
          this.view = true;
        }
      } else {
        this.view = false;
        this.router.navigateByUrl("/login")
      }
    })


  }
}
