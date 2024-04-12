import { SocialAuthService } from '@abacritt/angularx-social-login';
import { Component, OnInit } from '@angular/core';
import { tick } from '@angular/core/testing';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { User } from '../../../model/user';
import { ActivatedRoute, Router } from '@angular/router';
import { UserService } from '../../../Services/user/user.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrl: './login.component.css'
})
export class LoginComponent implements OnInit {

  login!: FormGroup;
  signup!: FormGroup;
  login_submitted = false;
  signup_submitted = false;
  login_swip = "";
  signup_disable = "";
  singin_disable = "";
  singin_active = "";
  singin_flip = "";
  singup_flip = "";
  filp = "";
  CurrentTime = new Date().toLocaleTimeString('en-US', {
    hour: 'numeric', minute: '2-digit', second: "2-digit", day: "2-digit", month: "2-digit", year: "2-digit", hour12: true
  }).toString();
  returnUrl!:string;
  constructor(private Formbuilder: FormBuilder,
     private googleauth: SocialAuthService,
     private activaterouter:ActivatedRoute,
     private userservice:UserService,
     private router:Router) { }
  ngOnInit(): void {
  this.returnUrl=this.activaterouter.snapshot.queryParams.returnUrl;

    this.signup = this.Formbuilder.group({
      name: ["", Validators.required],
      email: ["", [Validators.email, Validators.required]],
      password: ["", [Validators.required]]
    });
    this.login = this.Formbuilder.group({
      email: ["", [Validators.required, Validators.email]],
      password: ["", [Validators.required]]
    });
    this.googleauth.authState.subscribe((user) => {
      console.log(user)
      const userx: User = {
        id: '',
        name: user.name,
        email: user.email,
        profile: user.photoUrl,
        password: "",
        type: 'google',
        login_date: this.CurrentTime,
        token: '',
        admin: false
      }
      this.userservice.Goolge_Login(userx).subscribe(_=>{
        this.router.navigateByUrl(this.returnUrl)
      })
    })
  }
  get Login_Control() {
    return this.login.controls;
  }
  get signup_control() {
    return this.signup.controls;
  }



  move() {
    if (this.login_swip) {
      this.login_swip = "";
      this.signup_disable = "";
      this.singin_disable = "";
      this.singin_active = "";
      this.singin_flip = "";
      this.singup_flip = "";
      this.filp = "";
    }
    else {
      this.login_swip = "login-active";
      this.signup_disable = "signup-disable";
      this.singin_disable = "singin-disable";
      this.singin_active = "signin_active";
      this.singin_flip = "signin_flip_disable";
      this.singup_flip = "signin_flip_active";
      this.filp = "flip"

    }
    console.log("x")
  }


  login_submit() {
    this.login_submitted=true
    if (this.login.invalid) {
      return;
    }
    const login = this.login.value;
    const user: User = {
      id: '',
      name: '',
      email: login.email,
      profile: '',
      password: login.password,
      type: 'email',
      login_date: this.CurrentTime,
      token: '',
      admin: false
    }
    this.userservice.Login(user).subscribe(_=>{
      this.router.navigateByUrl(this.returnUrl)
    })
  }
  signup_submit() {
    this.signup_submitted=true
    if (this.signup.invalid) {
      return;
    }
    const signup = this.signup.value;

    const user: User = {
      id: '',
      name: signup.name,
      email: signup.email,
      profile: '',
      password: signup.password,
      type: 'email',
      login_date: this.CurrentTime,
      token: '',
      admin: false
    }
    this.userservice.Signup(user).subscribe(_=>{
      this.router.navigateByUrl(this.returnUrl)
    })
  }
}
