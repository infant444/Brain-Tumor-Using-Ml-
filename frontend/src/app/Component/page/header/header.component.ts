import { Component,HostListener,OnInit } from '@angular/core';
import { UserService } from '../../../Services/user/user.service';
import { User } from '../../../model/user';
import { Router } from '@angular/router';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent implements OnInit {
  profile_url!:string;
  headerfixed=false;
  user!:User;
  pop_up_nav=false;
  constructor(private userserviecs:UserService,private router:Router){}
  ngOnInit(): void {
    this.profile_url="../../../../assets/images/profile.png"
    this.userserviecs.userObeservable.subscribe((user)=>{
      this.user=user;
      if(user.profile!=""){
        this.profile_url=user.profile;
      }
      else{
    this.profile_url="../../../../assets/images/profile.png"
      }
    })
  }
  @HostListener('window:scroll', ['$event'])
  scroll(){
    if (window.scrollY > 82.2) {
      this.headerfixed = true;
    }
     else{
      this.headerfixed = false;
    }

  }
  profile() {
    this.router.navigateByUrl('/profile')
    }
  navchang(){
    if(this.pop_up_nav){
      this.pop_up_nav=false;
    }else{
      this.pop_up_nav=true;
    }
  }
}
